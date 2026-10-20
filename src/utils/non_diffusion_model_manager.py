import torch
from ..configs.configs_base import configs as configs_base
from ..configs.configs_data import data_configs
from ..configs.configs_inference import inference_configs
from ..protenix.config import parse_configs
from ..protenix.data.infer_data_pipeline import get_inference_dataloader
from ..runner.inference import InferenceRunner
from ..protenix.utils.torch_utils import to_device
from .io import AMINO_ACID_ATOMS_ORDER, ATOM_NAME_TO_ELEMENT, create_atom_mask, SEQUENCE_TYPE_TO_ATOM_DICTIONARY, SEQUENCE_TYPE_TO_RESIDUE_KIND
from ..protenix.data.utils import save_structure_cif
from tqdm import tqdm
import os 
from copy import deepcopy
import numpy as np
from ..protenix.model.sample_confidence import logits_to_score
import sys
import gemmi
from .io import load_pdb_atom_locations, load_pdb_atom_locations_full
from ..protenix.metrics.rmsd import self_aligned_rmsd


def compare_dict(d_1, d_2):
    for key in d_1.keys():
        if not (d_1[key] == d_2[key]).all:
            return False
    return True
    
def clip_norm(tensor, norm):
    with torch.no_grad():
        tensor_norm = tensor.norm(dim=-1, keepdim=True)
        norm = norm.reshape(tensor_norm.shape)
        clipping_tensor = torch.ones_like(norm)
        clipping_mask = tensor_norm > norm
        clipping_tensor[clipping_mask] = norm[clipping_mask] / tensor_norm[clipping_mask]
        tensor.data *= clipping_tensor
    return tensor

class MSA:
    """
        This is a basic class that contains the 3 tensors that represent MSA embeddings
    """
    def __init__(self, s_inputs: torch.Tensor, s: torch.Tensor, z: torch.Tensor):
        self.s_inputs = s_inputs
        self.s = s
        self.z = z
    
    def __iter__(self):
        yield self.s_inputs
        yield self.s
        yield self.z
    
    def parameters(self):
        parameters = []
        for tensor in self:
            if tensor.requires_grad:
                parameters.append(tensor)
        return parameters     
    
    def zeros_like(self):
        return MSA(torch.zeros_like(self.s_inputs), torch.zeros_like(self.s), torch.zeros_like(self.z))
    
    def clone(self):
        return MSA(self.s_inputs.clone(), self.s.clone(), self.z.clone())
    
    def to(self, device):
        self.s_inputs = self.s_inputs.to(device)
        self.s = self.s.to(device)
        self.z = self.z.to(device)
    
    def requires_grad(self, required_grad, apply_to_s_inputs=True):
        self.s_inputs.requires_grad = apply_to_s_inputs and required_grad
        self.s.requires_grad = required_grad
        self.z.requires_grad = required_grad

    def detach(self):
        self.s_inputs = self.s_inputs.detach()
        self.s = self.s.detach()
        self.z = self.z.detach()
        return self
    
    def __add__(self, other):
        return MSA(*[item_1 + item_2 for item_1, item_2 in zip(self, other)])

def atom_mask_from_residue_mask(atom_array, residue_mask):
    activated_residues = set(residue_mask.nonzero().squeeze().tolist())
    atom_mask = torch.zeros(atom_array.shape[0], dtype=torch.bool, device="cpu")
    for i, atom in enumerate(atom_array):
        if (atom.res_id - 1) in activated_residues:
            atom_mask[i] = True
    atom_mask = atom_mask.to(residue_mask.device)
    return atom_mask

class ProtenixModelManager:
    def __init__(self, sequences_dictionary, pdb_id,
                should_align_to_chains=[0],
                assembly_identifier=None,
                chains_to_read=[0],
                ROI_residues = None,
                reference_pdb=None,
                pdb_contains_missing_atoms=False,
                N_cycle=10, chunk_size=256, diffusion_N=200,
                gamma0=0.8,
                gamma_min=1.0,
                noise_scale_lambda=1.003,
                step_scale_eta = 1,
                dtype="fp32",
                use_deepspeed_evo_attention=False,
                msa_save_dir="./alignment_dir",
                msa_embedding_cache_dir="./msa_cache",
                pairformer_mixed_precision=False,
                model_checkpoint_path = "./src/af3-dev/release_model/model_v1.pt",
                dump_dir = "./output",
                use_msa=True,
                batch_size=1,
                device="cuda" if torch.cuda.is_available() else "cpu",
                ligands=None,
                covalent_bonds=None,
                ):
        self.pdb_id = pdb_id
        self.reference_pdb = reference_pdb
        self.should_align_to_chains = should_align_to_chains
        self.sequences_dictionary = sequences_dictionary
        self.ligands = deepcopy(ligands) if ligands is not None else []
        self.covalent_bonds = deepcopy(covalent_bonds) if covalent_bonds is not None else []
        self.assembly_identifier = assembly_identifier
        self.pdb_contains_missing_atoms = pdb_contains_missing_atoms
        self.full_sequences = [[dictionary["sequence"],]*dictionary["count"] for dictionary in sequences_dictionary]
        # flattening the list of lists
        self.full_sequences = [item for sublist in self.full_sequences for item in sublist]
        self.chains_to_read = chains_to_read
        self.ROI_residues = ROI_residues
        self.sequence_types = [
            sequence_type
            for dictionary in sequences_dictionary
            for sequence_type in [dictionary.get("sequence_type", "proteinChain")] * dictionary["count"]
        ]

        if reference_pdb is None:
            # Sequence-only mode (e.g. user-supplied NMR restraints with no deposited
            # entry). Topology comes from the Protenix dataloader, which is built from
            # the sequences, so the reference was only ever sizing the masks below.
            self.reference_atom_locations = None
            n_atoms_from_sequence = create_atom_mask(
                self.full_sequences,
                [[] for _ in self.full_sequences],
                sequence_types=self.sequence_types,
            ).shape[0]
            self.resolved_pdb_to_full_mask = torch.ones(n_atoms_from_sequence, dtype=torch.bool, device=device)
        elif self.pdb_contains_missing_atoms:
            self.reference_atom_locations, self.resolved_pdb_to_full_mask = load_pdb_atom_locations_full(
                reference_pdb,
                full_sequences = self.full_sequences,
                chains_to_read=chains_to_read,
                sequence_types=self.sequence_types,
            )
            self.resolved_pdb_to_full_mask = self.resolved_pdb_to_full_mask.to(device)
        else:
            self.reference_atom_locations = load_pdb_atom_locations(reference_pdb)
            self.resolved_pdb_to_full_mask = torch.ones(self.reference_atom_locations.shape[1], dtype=torch.bool, device=self.reference_atom_locations.device)

        # TODO: Advaith
        if self.ROI_residues is not None:
            self.ROI_atom_mask = create_atom_mask(self.full_sequences, ROI_residues, sequence_types=self.sequence_types).to(device)
        else:
            self.ROI_atom_mask = torch.zeros_like(self.resolved_pdb_to_full_mask, dtype=torch.bool)

        if self.reference_atom_locations is not None:
            self.reference_atom_locations = self.reference_atom_locations.to(device)

        self.N_cycle = N_cycle
        self.diffusion_N = diffusion_N
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.use_deepspeed_evo_attention = use_deepspeed_evo_attention
        self.msa_save_dir = msa_save_dir
        self.msa_embedding_cache_dir = msa_embedding_cache_dir

        if self.assembly_identifier is not None:
            self.msa_full_embedding_cache_dir = os.path.join(self.msa_embedding_cache_dir, self.pdb_id, self.assembly_identifier)
        else:
            self.msa_full_embedding_cache_dir = os.path.join(self.msa_embedding_cache_dir, self.pdb_id)

        self.pairformer_mixed_precision = pairformer_mixed_precision
        self.model_checkpoint_path = model_checkpoint_path
        self.dump_dir = dump_dir
        self.gamma0 = gamma0
        self.gamma_min = gamma_min
        self.noise_scale_lambda = noise_scale_lambda
        self.step_scale_eta = step_scale_eta
        self.use_msa = use_msa
        self.batch_size = batch_size

        self.atom_array = None
        self.input_feature_dict = None
        self.eval_data_dict = None

        self.msa = None

        self.s_init = None
        self.z_init = None
        self.s_inputs_init = None

        self.protenix_model = None
        self.denoise_net = None
        self.noise_schedule = None

        self.device = device
        self._setup(device)
    
    def align_models_to_reference(self, strucutres, reduced_atom_mask=None):
        if self.reference_atom_locations is None:
            raise RuntimeError(
                "align_models_to_reference() requires a reference structure, but this "
                "ProtenixModelManager was built without one (reference_pdb=None). "
                "Supply a reference PDB if alignment is needed."
            )
        if reduced_atom_mask is not None:
            atom_mask = reduced_atom_mask
        else:
            atom_mask = torch.ones(strucutres.shape[1], device=strucutres.device, dtype=torch.bool)[None]

        alignment_mask = (
            atom_mask
            & self.resolved_pdb_to_full_mask.to(atom_mask.device)
            & ~self.ROI_atom_mask.to(atom_mask.device)
            )
        if not alignment_mask.any():
            raise ValueError("No reference atoms are available for alignment.")
        
        _, aligned_structure, _, _ = self_aligned_rmsd(strucutres, self.reference_atom_locations, alignment_mask,)
        return aligned_structure

    def _generate_msa_configuration(self):
        entities = []
        
        for i, sequence_dict in enumerate(self.sequences_dictionary):
            sequence_type = sequence_dict.get("sequence_type", "proteinChain")
            polymer = {
                "sequence": sequence_dict["sequence"], 
                "count": sequence_dict["count"], 
                "modifications": deepcopy(sequence_dict.get("modifications", [])),
                }
            if sequence_type == "proteinChain":
                polymer["msa"] = {
                    "precomputed_msa_dir": os.path.join(self.msa_save_dir, "msa", str(i + 1)), 
                    "pairing_db": "uniref100",
                    }
            
            entities.append({sequence_type: polymer})
        
        # ligands
        entities.extend(
            {"ligand": deepcopy(ligand)} 
            for ligand in self.ligands
            )
        
        job = {
            "sequences": entities, 
            "modelSeeds": [], 
            "assembly_id": (
                "1" if self.assembly_identifier is None 
                else self.assembly_identifier
                ), 
            "name": self.pdb_id
            }
        
        if self.covalent_bonds:
            job["covalent_bonds"] = deepcopy(self.covalent_bonds)

        return [job]

    def _generate_configs(self, device=None):
        configs = {**configs_base, **{"data": data_configs}, **inference_configs}
        configs.update(
            {
                "load_checkpoint_path": self.model_checkpoint_path,
                "dump_dir": self.dump_dir,
                "use_deepspeed_evo_attention": self.use_deepspeed_evo_attention,
            }
        )
        configs["dtype"] = self.dtype
        configs["use_msa"] = self.use_msa
        if device is not None:
            configs["device"] = device
        configs["model"]["N_cycle"] = self.N_cycle
        configs["sample_diffusion"]["N_step"] = self.diffusion_N
        configs["sample_diffusion"]["N_sample"] = self.batch_size
        
        # parse_configs reads argv inside, and this breaks things because we want it to be empty, we we will cache it
        argv = sys.argv[:]
        sys.argv = argv[:1]
        configs = parse_configs(
            configs=configs,
            fill_required_with_null=True,
        )
        sys.argv = argv
        # this line makes the diffusion process run in parallel
        configs.infer_setting.sample_diffusion_chunk_size = None
        return configs
    
    def _setup_s_init_z_init(self, inplace_safe=True):
        
        # self.protenix_model.input_embedder.eval()
        # self.protenix_model.template_embedder.eval()
        # self.protenix_model.msa_module.eval()
        # self.protenix_model.pairformer_stack.eval()

        # Line 1-5
        s_inputs = self.protenix_model.input_embedder(
            self.input_feature_dict, inplace_safe=False, chunk_size=self.chunk_size
        )  # [..., N_token, 449]
        s_init = self.protenix_model.linear_no_bias_sinit(s_inputs)
        z_init = (
            self.protenix_model.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.protenix_model.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  #  [..., N_token, N_token, c_z]
        if inplace_safe:
            z_init += self.protenix_model.relative_position_encoding(self.input_feature_dict)
            z_init += self.protenix_model.linear_no_bias_token_bond(
                self.input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        else:
            z_init = z_init + self.protenix_model.relative_position_encoding(self.input_feature_dict)
            z_init = z_init + self.protenix_model.linear_no_bias_token_bond(
                self.input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        self.s_init = s_init
        self.z_init = z_init
        self.s_inputs_init = s_inputs
    
    def _load_cached_msa_embedding(self): # not configuration anymore, the prepared msa embeddings generated using msa configs etc...! 
        if self.ligands or self.covalent_bonds:
            return
        file_path = os.path.join(self.msa_full_embedding_cache_dir, f"msa.pt")
        if os.path.exists(file_path):
            self.msa = torch.load(file_path, weights_only=False)
            self.msa.to(self.device)
    
    def _save_cache_msa_embedding(self):
        if self.ligands or self.covalent_bonds:
            return
        file_path = os.path.join(self.msa_full_embedding_cache_dir, f"msa.pt")
        if not os.path.exists(file_path):
            os.makedirs(self.msa_full_embedding_cache_dir, exist_ok=True)
            self.msa.to("cpu")
            torch.save(self.msa, file_path)
            self.msa.to(self.device)

    def _expand_masks_to_model_topology(self):
        """
        Place existing polymer masks onto the complete model atom array.
        """
        device = self.device
        n_atoms = len(self.atom_array)
        
        polymer_entity_ids = {str(i + 1) for i in range(len(self.sequences_dictionary))}
        polymer_mask_np = np.isin(self.atom_array.label_entity_id, list(polymer_entity_ids),)
        polymer_indices = torch.as_tensor(np.flatnonzero(polymer_mask_np), dtype=torch.long, device=device,)
        n_polymer_atoms = len(polymer_indices)
        
        self.polymer_atom_mask = torch.as_tensor(polymer_mask_np, dtype=torch.bool, device=device)
        self.ligand_atom_mask = ~self.polymer_atom_mask
        
        # ROI for polymers only
        old_roi = self.ROI_atom_mask.to(device=device, dtype=torch.bool)
        if old_roi.numel() != n_polymer_atoms:
            raise ValueError(
                f"Polymer ROI mask has {old_roi.numel()} atoms, but "
                f"Protenix constructed {n_polymer_atoms} polymer atoms. "
                "Check the sequence and polymer topology."
                )
            
        self.ROI_atom_mask = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        self.ROI_atom_mask[polymer_indices] = old_roi
        
        if self.reference_atom_locations is None:
            # No reference comparison: all generated atoms are available.
            self.resolved_pdb_to_full_mask = torch.ones(n_atoms, dtype=torch.bool, device=device)
            return
        
        old_resolved = self.resolved_pdb_to_full_mask.to(device=device, dtype=torch.bool)
        if old_resolved.numel() != n_polymer_atoms:
            raise ValueError(
                "The reference mask does not match the model's polymer "
                "atom count. Reference atom mapping must be checked."
                )
        
        resolved = torch.zeros(n_atoms, dtype=torch.bool, device=device)
        resolved[polymer_indices] = old_resolved
        
        reference = self.reference_atom_locations.to(device)
        expanded_reference = reference.new_zeros(reference.shape[:-2] + (n_atoms, 3))
        
        if reference.shape[-2] == n_polymer_atoms:
            expanded_reference[..., polymer_indices, :] = reference
        elif reference.shape[-2] == int(old_resolved.sum().item()):
            matched_indices = polymer_indices[old_resolved]
            expanded_reference[..., matched_indices, :] = reference
        else:
            raise ValueError(
                "Reference coordinates do not match either the complete "
                "polymer topology or its resolved atoms."
                )
        
        self.reference_atom_locations = expanded_reference
        self.resolved_pdb_to_full_mask = resolved

    def _setup(self, device=None):
        configs = self._generate_configs(device)
        dataloader = get_inference_dataloader(configs=configs, msa_configuration=self._generate_msa_configuration())
        self.eval_data_dict, self.atom_array, error_msg = next(iter(dataloader))[0]
        if error_msg:
            raise RuntimeError(f"Inference data featurization failed, {error_msg}")
        
        output_names = np.asarray(self.atom_array.res_name, dtype="U64").copy()
        
        for ligand_index, ligand in enumerate(self.ligands, start=1):
            specification = ligand["ligand"]
            
            if not specification.startswith("FILE_") and specification[5:].lower().endswith(".sdf"):
                continue
            with open(specification[5:]) as handle:
                title = handle.readline().strip()
            # Use CCD title for residue names.
            if not title:
                raise ValueError("Missing SDF title, which is used as the ligands residue "
                                 "name and is usually the CCD ID.")
            if len(title) <= 3 and not any(character.isspace() for character in title):
                output_name = title

            entity_id = str(len(self.sequences_dictionary) + ligand_index)
            entity_mask = self.atom_array.label_entity_id == entity_id
            output_names[entity_mask] = output_name
            
        self.atom_array.set_annotation("output_res_name", output_names)
        
        self._expand_masks_to_model_topology()
        self.runner = InferenceRunner(configs)
        device = self.runner.device
        self.protenix_model = self.runner.model
        self.denoise_net = self.protenix_model.diffusion_module
        self.noise_schedule = self.protenix_model.inference_noise_scheduler(
            N_step=self.diffusion_N, device=device, dtype=torch.float32
        )
        input_feature_dict = to_device(self.eval_data_dict["input_feature_dict"], device)
        self.input_feature_dict = input_feature_dict

        self._load_cached_msa_embedding()
        if self.msa is None:
            with torch.no_grad():
                self._setup_s_init_z_init()

                s = torch.zeros_like(self.s_init)
                z = torch.zeros_like(self.z_init)
                # saving msa because we need self.msa.s_inputs from the pairformer_cycle
                self.msa = MSA(self.s_inputs_init, s, z)

                # some oligomeric msa and pairformers are just too big..!
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16 if self.pairformer_mixed_precision else torch.float32):
                    if self.pairformer_mixed_precision: 
                        s = s.to(torch.bfloat16)
                        z = z.to(torch.bfloat16)
                    for _ in tqdm(range(self.N_cycle), desc="Pairformer Cycle"):
                        s,z = self.pairformer_cycle(s,z)
                if self.pairformer_mixed_precision:
                    s = s.to(torch.float32)
                    z = z.to(torch.float32)
                self.msa = MSA(self.s_inputs_init, s, z)
                self._save_cache_msa_embedding()
    
    def pairformer_cycle(self, s, z, s_inputs=None, s_init=None, z_init=None, input_feature_dict=None, inplace_safe=False):
        s_inputs = s_inputs if s_inputs is not None else self.msa.s_inputs
        s_init = s_init if s_init is not None else self.s_init
        z_init = z_init if z_init is not None else self.z_init
        input_feature_dict = input_feature_dict if input_feature_dict is not None else self.input_feature_dict

        z = z_init + self.protenix_model.linear_no_bias_z_cycle(self.protenix_model.layernorm_z_cycle(z))
        if inplace_safe and self.protenix_model.template_embedder.n_blocks > 0:
            z += self.protenix_model.template_embedder(
                input_feature_dict,
                z,
                use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                use_deepspeed_evo_attention=False,
                use_lma=self.protenix_model.configs.use_lma,
                inplace_safe=inplace_safe,
                chunk_size=self.chunk_size,
            )
        elif self.protenix_model.template_embedder.n_blocks > 0:
            z = z + self.protenix_model.template_embedder(
                input_feature_dict,
                z,
                use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                use_deepspeed_evo_attention=False,
                use_lma=self.protenix_model.configs.use_lma,
                inplace_safe=inplace_safe,
                chunk_size=self.chunk_size,
            )
        z = self.protenix_model.msa_module(
            input_feature_dict,
            z,
            s_inputs,
            pair_mask=None,
            use_memory_efficient_kernel=self.protenix_model.configs.use_memory_efficient_kernel,
            use_deepspeed_evo_attention=False,
            use_lma=self.protenix_model.configs.use_lma,
            inplace_safe=inplace_safe,
            chunk_size=self.chunk_size,
        )
        s = s_init + self.protenix_model.linear_no_bias_s(self.protenix_model.layernorm_s(s))
        s, z = self.protenix_model.pairformer_stack(
            s,
            z,
            pair_mask=None,
            use_memory_efficient_kernel=self.protenix_model.configs.use_memory_efficient_kernel,
            use_deepspeed_evo_attention=False,
            use_lma=self.protenix_model.configs.use_lma,
            inplace_safe=inplace_safe,
            chunk_size=self.chunk_size,
        )
        return s, z

    def get_confidance_scores(self, structures, inplace_safe=False):
        plddt_pred, pae_pred, pde_pred, resolved_pred =  self.protenix_model.run_confidence_head(
            input_feature_dict=self.input_feature_dict,
            s_inputs=self.msa.s_inputs,
            s_trunk=self.msa.s,
            z_trunk=self.msa.z,
            pair_mask=None,
            x_pred_coords=structures,
            use_memory_efficient_kernel=self.protenix_model.configs.use_memory_efficient_kernel,
            use_deepspeed_evo_attention=False,
            use_lma=self.protenix_model.configs.use_lma,
            inplace_safe=inplace_safe,
            chunk_size=self.chunk_size,
        )
        score = logits_to_score(plddt_pred, 0, 100, 50)
        return score

    def get_t_hat(self, start_index=0, end_index=None):
        """
        end_index to support recycling steps
        """
        if end_index is None:
            end_index = start_index + 1
        c_tau_last, c_tau = self.noise_schedule[[start_index, end_index]]
        gamma = float(self.gamma0) if c_tau > self.gamma_min else 0
        t_hat = c_tau_last * (gamma + 1)
        return t_hat
    
    def get_x_start(self, batch_size=1, number_of_atoms=None):
        number_of_atoms = number_of_atoms or self.atom_array.shape[0]
        return self.noise_schedule[0] * torch.randn(
            size=(batch_size, number_of_atoms, 3), device=self.runner.device, dtype=torch.float32
        )
    
    def get_x_noisy(self, x_t, start_index=0, end_index=None):
        if end_index is None:
            end_index = start_index + 1

        if end_index > start_index:
            c_tau_last = self.noise_schedule[end_index]
            t_hat = self.get_t_hat(end_index)
            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_t + self.noise_scale_lambda * delta_noise_level * torch.randn(
                    size=x_t.shape, device=x_t.device, dtype=torch.float32
                )
            return x_noisy
        else:
            # Take forward steps
            sigma_start = self.noise_schedule[start_index]
            sigma_end = self.noise_schedule[end_index]
            std = (sigma_end**2 - sigma_start**2).sqrt()
            eps = torch.randn(size=x_t.shape, device=x_t.device, dtype=torch.float32)
            return x_t + self.noise_scale_lambda * std * eps

    # def get_x_t_from_x_0_hat(self, x_noisy, x_0_hat, start_index, end_index=None, guidance_direction=None, step_size=1.0, normalize_gradients=False):
    def get_x_t_from_x_0_hat(
            self, x_noisy, x_0_hat, start_index, end_index=None, 
            guidance_direction=None, step_size=1.0, normalize_gradients=False, normalize_by_current_gradient_strength=False,
            structures_gradient_norm=1.0, guidance_scale_gradually_increase=False,
        ):
        if end_index is None:
            end_index = start_index + 1
        is_backward_step = end_index > start_index

        if is_backward_step:
            c_tau = self.noise_schedule[end_index]
            t_hat = self.get_t_hat(start_index)
            delta = (x_noisy - x_0_hat) / t_hat[..., None, None]
            if guidance_direction is not None:
                if normalize_gradients:
                    guidance_direction = guidance_direction * delta.flatten(1,-1).norm(dim=-1)[:, None, None]
                delta = delta + step_size * guidance_direction
            dt = c_tau - t_hat
            x_l = x_noisy + self.step_scale_eta * dt[..., None, None] * delta
            return x_l
        else:
            # Also, applying no guidance when adding the noise back (no need, will be applied during denoising anyway..!)
            return x_noisy
    
    def get_x_0_hat_from_x_noisy(self, x_noisy, t_hat=None, start_index=None,msa=None,input_feature_dict=None, inplace_safe=False):
        # make sure that t_hat is not None or t is not None, but not both
        assert not ((t_hat is None) and (start_index is None))
        assert not ((not t_hat is None) and (not start_index is None))
        if t_hat is None:
            t_hat = self.get_t_hat(start_index)
        msa = msa or self.msa
        input_feature_dict = input_feature_dict or self.input_feature_dict
        return self.denoise_net(
                    x_noisy=x_noisy,
                    t_hat_noise_level=t_hat.reshape([1]).repeat(x_noisy.shape[0]),
                    input_feature_dict=input_feature_dict,
                    s_inputs=msa.s_inputs,
                    s_trunk=msa.s,
                    z_trunk=msa.z,
                    chunk_size=self.chunk_size,
                    inplace_safe=inplace_safe,
                )
    
    def single_diffusion_step(self, x_t, start_index,end_index=None, inplace_safe=True, hook=None):
        x_noisy = self.get_x_noisy(x_t, start_index)
        x_0_hat = self.get_x_0_hat_from_x_noisy(x_noisy, start_index=start_index, inplace_safe=inplace_safe)
        if hook is not None:
            hook(x_0_hat, start_index)
        x_t_end = self.get_x_t_from_x_0_hat(x_noisy, x_0_hat, start_index,end_index=end_index)
        return x_t_end
                
    def save_structure_cif(self, structure, file_name):
        polytype = self.eval_data_dict["entity_poly_type"]
        dir_path = os.path.dirname(file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        save_structure_cif(self.atom_array, structure, file_name, polytype, self.pdb_id)

    def save_structure_pdb(self, structure, atom_array, write_file_name):
        gemmi_structure = gemmi.read_structure(self.reference_pdb)
        chain = gemmi_structure[0][0]
        num_structures = len(gemmi_structure)
        for i in range(num_structures - 1, 0, -1):
            del gemmi_structure[i]
        num_models = len(gemmi_structure[0])
        for i in range(num_models - 1, 0, -1):
            del gemmi_structure[0][i]
        atom_index = 0
        structure = structure.cpu().detach().numpy()
        for res in chain:
            res_name = res.name
            atom_order = AMINO_ACID_ATOMS_ORDER[res_name]
            atoms = [[atom for atom in res if atom.name == atom_name][0] for atom_name in atom_order]
            if res[-1].name == "OXT":
                atoms.append(res[-1])
            for atom in atoms:
                atom.pos = gemmi.Position(*structure[atom_index])
                atom_index += 1

        os.makedirs(os.path.dirname(write_file_name), exist_ok=True)
        gemmi_structure.write_pdb(write_file_name)


def save_structure_full(structure, full_sequences, sequence_types, atom_array, write_file_name, 
                        bfactors=None, atom_mask=None, start_residue_from=1):
    
    def to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asanyarray(value)
    
    xyz = to_numpy(structure)
    n_atoms = len(atom_array)
    
    if xyz.shape != (n_atoms, 3):
        raise ValueError(
            f"Expected coordinates ({n_atoms}, 3), "
            f"received {xyz.shape}."
            )
    if not np.isfinite(xyz).all():
        raise ValueError("Predicted coordinates contain NaN or infinity.")
        
    keep = np.ones(n_atoms, dtype=bool) if atom_mask is None else to_numpy(atom_mask).astype(bool)
    b_values = np.zeros(n_atoms) if bfactors is None else to_numpy(bfactors)
    if keep.shape != (n_atoms,) or b_values.shape != (n_atoms,):
        raise ValueError("Atom_mask/B-factors must match the model atom count.")

    extenstion = os.path.splitext(write_file_name)[1].lower()
    if extenstion not in (".pdb", ".cif", ".mmcif"):
        raise ValueError("Output extension must be .pdb, .cif, or .mmcif.")
        
    result = gemmi.Structure()
    result.name = "prediction"
    model = gemmi.Model("1")
    
    chains = {}
    residues = {}
    addresses = {}
    serials = {}
    residue_keys = {}
    offset = int(start_residue_from or 1) - 1
    
    for index in np.flatnonzero(keep):
        chain_id = str(atom_array.label_asym_id[index])
        
        if "output_res_name" in atom_array.get_annotation_categories():
            res_name = str(atom_array.output_res_name[index])
        else:
            res_name = str(atom_array.res_name[index])
        
        atom_name = str(atom_array.atom_name[index])
        hetero = bool(atom_array.hetero[index])
        
        res_number = int(atom_array.res_id[index])
        if not hetero:
            res_number += offset
        
        insertion_code = str(atom_array.ins_code[index]).strip() or " "
        key = (chain_id, res_number, insertion_code, res_name, hetero)
  
        if chain_id not in chains:
            chains[chain_id] = gemmi.Chain(chain_id)
            
        if key not in residues:
            residue = gemmi.Residue()
            residue.name = res_name
            residue.seqid = gemmi.SeqId(res_number, insertion_code)
            residue.het_flag = "H" if hetero else "A"
            residue.entity_type = gemmi.EntityType.NonPolymer if hetero else gemmi.EntityType.Polymer
            residues[key] = residue
            
        atom = gemmi.Atom()
        atom.name = atom_name
        atom.element = gemmi.Element(str(atom_array.element[index]))
        atom.pos = gemmi.Position(*map(float, xyz[index]))
        atom.occ = 1.0
        atom.b_iso = float(b_values[index])
        atom.serial = len(serials) + 1
        
        residues[key].add_atom(atom)
        serials[int(index)] = atom.serial
        residue_keys[int(index)] = key
        addresses[int(index)] = gemmi.AtomAddress(
            chain_id, 
            gemmi.SeqId(res_number, insertion_code),
            res_name, atom_name,
                                                  )
    
    for key, residue in residues.items():
        chains[key[0]].add_residue(residue)
    for chain in chains.values():
        model.add_chain(chain)
    result.add_model(model)
    result.setup_entities()
    
    retained_bonds = []
    ligand_component_bonds = set()
    if atom_array.bonds is not None:
        for index_1, index_2, _ in atom_array.bonds.as_array():
            index_1, index_2 = int(index_1), int(index_2)
            if index_1 not in serials or index_2 not in serials:
                continue
            # every topology bond is added to PDB CONNECT record
            retained_bonds.append((index_1, index_2))
            
            name_1 = str(atom_array.atom_name[index_1])
            name_2 = str(atom_array.atom_name[index_2])
            # Record explicit inter-residue topology connections.
            different_residues = residue_keys[index_1] != residue_keys[index_2]
            is_disulfide = (different_residues
                            and str(atom_array.res_name[index_1]) == "CYS"
                            and str(atom_array.res_name[index_2]) == "CYS"
                            and name_1 == "SG" and name_2 == "SG")
            
            has_ligand_atom = bool(atom_array.hetero[index_1]) or bool(atom_array.hetero[index_2])
            
            if different_residues and (is_disulfide or has_ligand_atom):
                connection = gemmi.Connection()
                connection.name = f"connection{len(result.connections) + 1}"
                connection.type = (
                    gemmi.ConnectionType.Disulf
                    if is_disulfide
                    else gemmi.ConnectionType.Covale
                                   )
                connection.asu = gemmi.Asu.Same
                connection.partner1 = addresses[index_1]
                connection.partner2 = addresses[index_2]
                connection.reported_distance = float(np.linalg.norm(xyz[index_1] - xyz[index_2]))
                result.connections.append(connection)
                
            if not different_residues and bool(atom_array.hetero[index_1]):
                name_1, name_2 = sorted((
                    str(atom_array.atom_name[index_1]),
                    str(atom_array.atom_name[index_2]),
                    ))
                ligand_component_bonds.add((
                    residue_keys[index_1][3],
                    name_1,
                    name_2,
                    ))
                
    os.makedirs(os.path.dirname(write_file_name) or '.', exist_ok=True)
    
    if extenstion == ".pdb":
        if not hasattr(result, "add_conect"):
            raise RuntimeError(
                "This Gemmi version lacks explicit CONNECT writing. "
                "Use mmCIF output or a Gemmi version with add_connect()."
                               )
        
        for index_1, index_2 in retained_bonds:
            result.add_conect(serials[index_1], serials[index_2], 1)
            
        options = gemmi.PdbWriteOptions()
        options.preserve_serial = True
        options.numbered_ter = False
        options.conect_records = True
        result.write_pdb(write_file_name, options)
    else:
        document = result.make_mmcif_document()
        
        if ligand_component_bonds:
            loop = document.sole_block().init_loop(
                "_chem_comp_bond.",
                ["comp_id", "atom_id_1", "atom_id_2", "value_order"],
                )
            for res_name, name_1, name_2 in sorted(ligand_component_bonds):
                # Connectivity is known; do not invent missing bond orders.
                loop.add_row([res_name, name_1, name_2, "?"])
        
        document.write_file(write_file_name)

