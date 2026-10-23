import numpy as np
import pandas as pd
import torch

from .abstract_loss_funciton import AbstractLossFunction
from ..utils.io import chain_segments_from_atom_array

class PRELossFunction(AbstractLossFunction):
    def __init__(
            self, 
            restraint_file,
            atom_array,
            protein_indices,
            protein_nmr,
            scale=1.0,
            ensemble_mode="r6",):
        self.atom_array = atom_array
        self.protein_nmr = protein_nmr
        self.protein_indices = protein_indices
        self.scale = float(scale)
        self.ensemble_mode = ensemble_mode
        self.last_value = None
        
        if not np.isfinite(self.scale) or self.scale < 0:
            raise ValueError("PRE scale must be finite and nonnegative.")
        if ensemble_mode not in {"r6", "mean"}:
            raise ValueError("PRE ensemble_mode must be 'r6' or 'mean'.")
        
        self.data = pd.read_csv(restraint_file, dtype={"chain1": str, "chain2": str},)
        self.data = self.data[self.data["type"].astype(str).str.upper() == "PRE"].reset_index(drop=True)
        
        self.has_restraints = not self.data.empty
        self.endpoints = []
        self.groups = []
        
        if not self.has_restraints:
            return
        
        required = {
            "chain1", "chain2", "constrain_id", 
            "residue1_num", "residue2_num", 
            "residue1_id", "residue2_id", 
            "atom1", "atom2", "lower_bound", "upper_bound", 
            }
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing PRE columns: {sorted(missing)}")
        
        if self.data[list(required)].isna().any().any():
            raise ValueError("PRE rows contain missing endpoint, ID, or bound values.")
        
        lower = pd.to_numeric(self.data.lower_bound.replace('.', 0)).to_numpy(float)
        upper = pd.to_numeric(self.data.upper_bound).to_numpy(float)
        
        if not (np.isfinite(lower).all() 
                and np.isfinite(upper).all() 
                and (lower >= 0).all() 
                and (upper >= lower).all()
                ):
            raise ValueError("PRE bounds must be finite, with 0 <= lower <= upper.")
        
        self.bounds = (lower, upper)
        
        labels, segments = chain_segments_from_atom_array(atom_array)
        self.chain_ranges = dict(zip(map(str, labels), segments))
        
        self.protein_chains = {str(label): index for index, label in enumerate(protein_nmr.chain_ids)}
        
        # Accept either the internal residue name or its output alias.
        if "output_res_name" in atom_array.get_annotation_categories():
            aliases = atom_array.output_res_name
        else:
            aliases = atom_array.res_name
        self.aliases = np.asarray(aliases).astype(str)
        
        for row_number, row in self.data.iterrows():
            self.endpoints.append(tuple(
                self._resolve_endpoint(row, side, row_number + 1) 
                for side in (1 ,2)
                ))
        
        for group_id, rows in self.data.groupby("constrain_id", sort=False):
            pairs = set(zip(rows.chain1, rows.chain2))
            if len(pairs) != 1:
                raise ValueError(f"PRE group {group_id} spans multiple chain pairs; "
                                 "use distinct constrain_id values for different pairs.")
            
            # As in the NOE loss, repeated IDs represent OR alternatives.
            self.groups.append((group_id, rows.index.tolist()))
            
    def _resolve_endpoint(self, row, side, row_number):
        chain = str(row[f"chain{side}"]).strip()
        
        if chain not in self.chain_ranges:
            raise ValueError(f"PRE row {row_number}: unknown chain {chain!r}; "
                             f"available chains: {list(self.chain_ranges)}")
            
        start, stop =self.chain_ranges[chain]
        number = float(row[f"residue{side}_num"])
        
        if not np.isfinite(number) or not number.is_integer():
            raise ValueError(f"PRE row {row_number}: residue number must be an integer.")
        
        resnum = int(number)
        
        # Author-numbering offsets apply only to protein endpoints.
        if chain in self.protein_chains:
            resnum -= self.protein_nmr.residue_offset
            
        resname = str(row[f"residue{side}_id"]).strip()
        atom = str(row[f"atom{side}"]).strip()
        indices = np.arange(start, stop)
        arr = self.atom_array
        
        residue = indices[
            (arr.res_id[indices] == resnum) 
            & (
                (arr.res_num[indices] == resname) 
                | (self.aliases[indices] == resname)
                )
            ]
        
        if len(residue) == 0:
            raise ValueError(f"PRE row {row_number}: residue "
                             f"{chain}:{int(number)}:{resname} was not found.")
        
        direct = residue[arr.atom_name[residue] == atom]
        
        if len(direct) == 1:
            return ("direct", int(direct[0]))
        
        if len(direct) > 1:
            raise ValueError(f"PRE row {row_number}: ambiguous atom "
                             f"{chain}:{resnum}:{atom}.")
            
        if chain in self.protein_chains and atom.startswith("H"):
            internal_name = str(arr.res_name[residue[0]])
            return ("hydrogen", self.protein_chains[chain], 
                    (resnum, internal_name, atom),)
        
        raise ValueError(f"PRE row {row_number}: atom {chain}:{resnum}:{atom} "
                         "is absent; ligand hydrogen generation is not implemented.")
        
    def _distances(self, coordinates):
        cache = None
        hydrogen_indicies = {}
        points = {}
        
        def point(endpoint):
            nonlocal cache
            
            if endpoint in points:
                return points[endpoint]
            
            if endpoint[0] == "direct":
                value = coordinates[:, endpoint[1], :]
            else:
                if cache is None:
                    indices = self.protein_indices.to(coordinates.device)
                    protein_coordinates = coordinates.index_select(-2, indices)
                    cache = self.protein_nmr._prepare_chain_cache(protein_coordinates)
                
                _, chain_index, name = endpoint
                
                if chain_index not in hydrogen_indicies:
                    hydrogen_indicies[chain_index] = self.protein_nmr._make_hname_index(
                        cache[chain_index].Hnames
                        )
                
                if name not in hydrogen_indicies[chain_index]:
                    raise ValueError("PRE protein hydrogen could not be generated: "
                                     f"{name}")
                
                hydrogen_index = hydrogen_indicies[chain_index][name]
                value = cache[chain_index].H[:, hydrogen_index, :]
            
            points[endpoint] = value
            return value
        
        distances = torch.stack([
            torch.linalg.vector_norm(
                point(left) - point(right), dim=-1
                ) 
            for left, right in self.endpoints
            ], dim=1)
        
        if not torch.isfinite(distances).all():
            raise ValueError("Nonfinite PRE endpoints distance.")
        
        return distances
    
    def _ensemble_distance(self, distances):
        if self.ensemble_mode == "mean":
            return distances.mean(dim=0, keepdim=True)
        
        safe = distances.clamp_min(1e-6)
        return safe.pow(-6).mean(dim=0, keepdim=True).pow(-1.0 / 6.0)
    
    def __call__(self, x_0_hat, time, structures=None, i=None, step=None):
        if not self.has_restraints:
            self.last_value = x_0_hat.sum() * 0.0
            return self.last_value, None
        
        distances = self._distances(x_0_hat)
        
        if not self.protein_nmr.iid_loss:
            distances = self._ensemble_distance(distances)
            
        lower = distances.new_tensor(self.bounds[0])
        upper = distances.new_tensor(self.bounds[1])
        
        violations = torch.relu(lower - distances) + torch.relu(distances - upper)
        
        # Select the smallest combined violation within each OR group.
        group_violations = torch.stack([
            violations[:, rows].amin(dim=1) 
            for _, rows in self.groups
            ], dim=1)
        
        self.last_value = group_violations.mean() * self.scale
        return self.last_value, None
    
    @torch.no_grad()
    def distance_restraint_report(self, coordinates, tolerance_A=0.0):
        if tolerance_A < 0:
            raise ValueError("tolerance_A must be nonnegative.")
            
        distances = self._distances(coordinates)
        values = torch.cat([distances, self._ensemble_distance(distances),], dim=0).cpu().numpy()
        
        labels = [f"member_{index}" for index in range(len(distances))]
        labels.append(f"ensemble_{self.ensemble_mode}_distance")
        
        records = []
        group_records = []
        summaries = []
        lower, upper = self.bounds
        
        for evaluation, current in zip(labels, values):
            lb = np.maximum(lower - current, 0)
            ub = np.maximum(current - upper, 0)
            
            for index, row in self.data.iterrows():
                records.append({
                    "evaluation": evaluation,
                    "pre_row": index + 1,
                    "constrain_id": row.constrain_id,
                    "chain1": row.chain1,
                    "chain2": row.chain2,
                    "endpoint1": (
                        f"{row.chain1}:{row.residue1_num}:"
                        f"{row.residue1_id}:{row.atom1}"
                        ),
                    "endpoint2": (
                        f"{row.chain2}:{row.residue2_num}:"
                        f"{row.residue2_id}:{row.atom2}"
                        ),
                    "lower_bound_A": lower[index],
                    "upper_bound_A": upper[index],
                    "distance_A": current[index],
                    "lower_violation_A": lb[index],
                    "upper_violation_A": ub[index],
                    "violation_A": lb[index] + ub[index],
                    "evaluated": True,
                    })
                
            group_values = []
            
            for group_id, rows in self.groups:
                best = min(rows, key=lambda k: (lb[k] + ub[k], ub[k], k),)
                violation = float(lb[best] + ub[best])
                group_values.append(violation)
                
                group_records.append({
                    "evaluation": evaluation,
                    "constrain_id": group_id,
                    "selected_pre_row": best + 1,
                    "alternatives_total": len(rows),
                    "alternatives_evaluated": len(rows),
                    "evaluated": True,
                    "distance_A": current[best],
                    "lower_violation_A": lb[best],
                    "upper_violation_A": ub[best],
                    "violation_A": violation,
                    "violated": violation > tolerance_A,
                    })
                
            group_values = np.asarray(group_values)
            violated = int((group_values > tolerance_A).sum())
            
            summaries.append({
                "evaluation": evaluation,
                "groups_total": len(self.groups),
                "groups_evaluated": len(self.groups),
                "groups_unresolved": 0,
                "groups_partially_resolved": 0,
                "groups_violated": violated,
                "fraction_violated_among_evaluated": violated / len(self.groups),
                "mean_violation_A": group_values.mean(),
                "max_violation_A": group_values.max(),
                "tolerance_A": tolerance_A,
                })
        
        return (pd.DataFrame(records), pd.DataFrame(group_records), pd.DataFrame(summaries),)
    

class NMRWithPRELoss(AbstractLossFunction):
    def __init__(self, protein_loss, pre_loss):
        self.protein_loss = protein_loss
        self.pre_loss = pre_loss
        self.last_value = None
        
    def __call__(self, x_0_hat, time, structures=None, i=None, step=None):
        kwargs = dict(structures=structures, i=i, step=step)
        
        protein_value, corrected = self.protein_loss(x_0_hat, time, **kwargs)
        pre_value, _ = self.pre_loss(x_0_hat, time, **kwargs)
        
        self.last_value = protein_value + pre_value
        return self.last_value, corrected
    
    def wandb_log(self, coordinates):
        result = dict(self.protein_loss.wandb_log(coordinates))
        
        result["pre_loss"] = None if self.pre_loss.last_value is None else float(self.pre_loss.last_value.detach().item())
        result["loss"] = None if self.last_value is None else float(self.last_value.detach().item())
        return result
    
    def post_optimization_step(self):
        self.protein_loss.post_optimization_step()
        
    def save_state(self, structures, folder_path, **kwargs):
        self.protein_loss.loss.save_state(self.protein_loss._select(structures),
                                          folder_path, **kwargs,)
        
    def distance_restraint_reports(self, coordinates, tolerance_A=0.0):
        reports = {}
        nmr = self.protein_loss.loss
        
        if not nmr.nmr_data.empty:
            reports["noe"] = nmr.distance_restraint_report(self.protein_loss._select(coordinates), 
                                                           tolerance_A=tolerance_A,)
            
        if self.pre_loss.has_restraints:
            reports["pre"] = self.pre_loss.distance_restraint_report(coordinates, tolerance_A=tolerance_A)
            
        return reports
            
