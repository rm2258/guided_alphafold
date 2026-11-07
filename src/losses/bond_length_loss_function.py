import numpy as np
import gemmi
import torch
from torch.utils.checkpoint import checkpoint

from .abstract_loss_funciton import AbstractLossFunction
from biotite.structure import connect_via_residue_names

class BondLengthLossFunction(AbstractLossFunction):
    def __init__(self, atom_array_object, device, collision_chunk_size=64):
        self._device = device
        self._atom_array_object = atom_array_object
        self._collision_chunk_size = int(collision_chunk_size)
        
        if self._collision_chunk_size < 1:
            raise ValueError("collision_chunk_size must be positive.")
        
        self._bonds, self._bond_lengths = self._initialize_topolgy_bonds()
        
        # store one radius per atom instead of an atom-by-atom matrix
        radii = np.array([gemmi.Element(atom.element).covalent_r
                          for atom in atom_array_object])
        self._atom_covalent_r = torch.as_tensor(radii, device=device)
        
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
        bond_lengths = (atom_locations[:, self._bonds[:, 0]] - atom_locations[:, self._bonds[:, 1]]).norm(dim=-1)
        
        errors = ((bond_lengths - self._bond_lengths).abs() - threshold).relu()
        return errors.pow(l).sum() / atom_locations.shape[0]
    
    def _collision_block(self, atom_locations, start, stop, padding):
        distances = (atom_locations[:, start:stop, None, :] - atom_locations[:, None, :, :]).norm(dim=-1)
        
        atom_count = atom_locations.shape[1]
        
        # The same mask applies to every ensemble member.
        mask = atom_locations.new_zeros((stop - start, atom_count))
        
        diagonal = torch.arange(start, stop, device=atom_locations.device)
        mask[diagonal - start, diagonal] = 3
        
        first = self._bonds[:, 0]
        second = self._bonds[:, 1]
        selected = (first >= start) & (first < stop)
        mask[first[selected] - start, second[selected]] = 3
        
        selected = (second >= start) & (second < stop)
        mask[second[selected] - start, first[selected]] = 3
        
        distances = distances + mask
        
        collision_distances = self._atom_covalent_r[start:stop, None] + self._atom_covalent_r[None, :]
        
        # Keep the original maximum across the ensemble
        violations = (collision_distances + padding - distances).max(dim=0)[0].relu()
        return violations.sum()
    
    def collision_loss(self, atom_locations, padding = 0.4):
        batch_size, atom_count = atom_locations.shape[:2]
        
        if batch_size == 0:
            raise ValueError("Collision loss requires at least one sample.")
            
        total = atom_locations.new_zeros((), dtype=self._atom_covalent_r.dtype)
        
        for start in range(0, atom_count, self._collision_chunk_size):
            stop = min(start + self._collision_chunk_size, atom_count)
            
            if torch.is_grad_enabled() and atom_locations.requires_grad:
                block_loss = checkpoint(self._collision_block, atom_locations, 
                                        start, stop, padding, use_reentrant=False)
            else:
                block_loss = self._collision_block(atom_locations, start, stop, padding)
                
            total = total + block_loss
        
        return total / batch_size
    
    def get_bond_loss(self, atom_locations):
        bond_loss = self.bond_length_loss(atom_locations)
        collision_loss = self.collision_loss(atom_locations)
        
        self._last_bond_length_loss = bond_loss.detach().item()
        self._last_collision_loss = collision_loss.detach().item()
        
        loss_values = bond_loss + collision_loss
        return loss_values 
    
    def __call__(self, x_0_hat, time, structures=None, i=None, step=None):
        return self.get_bond_loss(x_0_hat), None
    
    def wandb_log(self, x_0_hat):
        return {"bond length loss": self._last_bond_length_loss, 
                "collision loss": self._last_collision_loss}
    
    