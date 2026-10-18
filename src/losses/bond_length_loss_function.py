from .abstract_loss_funciton import AbstractLossFunction
import torch
from biotite.structure import connect_via_residue_names
import numpy as np
import gemmi

class BondLengthLossFunction(AbstractLossFunction):
    def __init__(self, atom_array_object, device):
        self._device = device
        self._atom_array_object = atom_array_object
        self._bonds, self._bond_lengths = self._initialize_topolgy_bonds()
        self._collision_distances = self._initialize_collision_distances()

        self._last_bond_length_loss = None
        self._last_collision_loss = None

    def _initialize_topolgy_bonds(self):
        topology = self._atom_array_object.bonds
        
        if topology is None:
            raise ValueError(
                "Bond-length loss requires the model's bond topology."
                )
        
        bonds = topology.as_array()
        
        if len(bonds):
            bonded_atoms = np.asarray(bonds[:, :2], dtype=np.int64)
        else:
            bonded_atoms = np.empty((0, 2), dtype=np.int64)
        
        bond_lengths = []
        for index_1, index_2 in bonded_atoms:
            element_1 = gemmi.Element(self._atom_array_object[index_1].element)
            element_2 = gemmi.Element(self._atom_array_object[index_2].element)
            bond_lengths.append(element_1.covalent_r + element_2.covalent_r)
        
        return (
            torch.as_tensor(bonded_atoms, device=self._device, dtype=torch.long,), 
            torch.as_tensor(bond_lengths, device=self._device, dtype=torch.float32,), 
                )
    
    def _initialize_collision_distances(self):
        atom_covalent_r = np.array([gemmi.Element(self._atom_array_object[index].element).covalent_r for index in range(len(self._atom_array_object))])
        atom_covalent_r = torch.tensor(atom_covalent_r, device=self._device)
        return (atom_covalent_r[None] + atom_covalent_r[:, None])
    
    def bond_length_loss(self, atom_locations, threshold=0.2 , l=2):
        bond_lengths = (atom_locations[:, self._bonds[...,0]] - atom_locations[:, self._bonds[...,1]]).norm(dim=-1)
        bond_lengths_diff = ((bond_lengths - self._bond_lengths).abs() - threshold).relu()
        return bond_lengths_diff.pow(l).sum() / atom_locations.shape[0]
    
    def collision_loss(self, atom_locations, padding = 0.4):
        distances = (atom_locations[:,:, None] - atom_locations[:,None]).norm(dim=-1)
        mask = torch.zeros_like(distances)
        mask[:, self._bonds[...,0], self._bonds[...,1]] = 3
        mask[:, self._bonds[...,1], self._bonds[...,0]] = 3
        diagonal = torch.arange(atom_locations.shape[1], device=atom_locations.device,)
        mask[:, diagonal, diagonal] = 3
        distances = distances + mask
        threshold_loss = (self._collision_distances + padding - distances).max(dim=0)[0].relu()
        return threshold_loss.sum() / atom_locations.shape[0]
    
    def get_bond_loss(self, atom_locations):
        bond_length_loss = self.bond_length_loss(atom_locations)
        collision_loss = self.collision_loss(atom_locations)
        loss_values = bond_length_loss + collision_loss
        self._last_bond_length_loss = bond_length_loss.item()
        self._last_collision_loss = collision_loss.item()
        return loss_values 
    
    def __call__(self, x_0_hat, time, structures=None, i=None, step=None):
        return self.get_bond_loss(x_0_hat), None
    
    def wandb_log(self, x_0_hat):
        return {"bond length loss": self._last_bond_length_loss, "collision loss": self._last_collision_loss}