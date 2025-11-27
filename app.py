import streamlit as st
import pandas as pd
import numpy as np
import itertools
from collections import Counter, defaultdict
import warnings
import base64

# RDKit Imports
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import RDLogger

# Visualization Imports
import plotly.express as px

# ------------------------------------------------------------------------------
# 0. 基础设置与美化配置
# ------------------------------------------------------------------------------

# 屏蔽 RDKit 日志
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

# 启用更优美的分子布局算法 (CoordGen)
rdDepictor.SetPreferCoordGen(True)

st.set_page_config(page_title="Polymer Fingerprint Explorer (Pro)", layout="wide")

# --- 高级配色方案 (RGB 0-1) ---
STYLE_COLS = {
    "backbone": (1.0, 0.42, 0.42),   # Coral Red
    "sidechain": (0.27, 0.72, 0.82), # Steel Blue
    "focus": (0.42, 0.80, 0.47),     # Emerald Green (中心/选中)
    "context": (1.0, 0.85, 0.24),    # Amber Yellow (邻居)
    "highlight_block": (0.2, 0.8, 0.6) # Teal (Block高亮)
}

def get_atom_mass_with_hs(atom):
    """计算原子质量，包含隐式氢"""
    mass = atom.GetMass()
    mass += atom.GetTotalNumHs() * 1.00794
    return mass

def draw_mol_svg(mol, highlight_atoms=None, highlight_colors=None, size=(1000, 600)):
    """
    生成美化后的 SVG 分子图
    """
    if mol is None: return ""
    
    try:
        rdDepictor.Compute2DCoords(mol)
    except:
        pass

    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    dopts = drawer.drawOptions()
    
    dopts.addAtomIndices = True          
    dopts.bondLineWidth = 3              
    dopts.highlightBondWidthMultiplier = 1 # Int类型
    dopts.clearBackground = False        
    dopts.fixedBondLength = 45           
    dopts.minFontSize = 14               
    
    if highlight_colors:
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_colors)
    elif highlight_atoms:
        drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
    else:
        drawer.DrawMolecule(mol)
        
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

# ------------------------------------------------------------------------------
# 1. 核心逻辑：Morphological Engine (M-Fingerprints)
# ------------------------------------------------------------------------------
class MorphologicalEngine:
    def __init__(self, mol, dummies=None):
        self.mol = mol
        self.valid = True
        if self.mol is None:
            self.valid = False
            return
        if dummies is None:
            self.dummies = [a.GetIdx() for a in self.mol.GetAtoms() if a.GetAtomicNum() == 0]
        else:
            self.dummies = dummies
        if len(self.dummies) != 2:
            self.valid = False
            return
            
        self.dist_matrix = Chem.GetDistanceMatrix(self.mol)
        self.ring_info = self.mol.GetRingInfo()
        self.atom_rings = self.ring_info.AtomRings()
        self.all_atom_indices = set([a.GetIdx() for a in self.mol.GetAtoms() if a.GetAtomicNum() != 0])
        self._classify_structure()

    def _classify_structure(self):
        try:
            self.path_tuple = Chem.rdmolops.GetShortestPath(
                self.mol, self.dummies[0], self.dummies[1]
            )
            self.path_atoms_set = set([idx for idx in self.path_tuple 
                                       if self.mol.GetAtomWithIdx(idx).GetAtomicNum() != 0])
        except:
            self.path_atoms_set = set(); self.valid = False; return

        self.backbone_expansion_atoms = set()
        self.backbone_rings_indices = []
        self.sidechain_rings_indices = []
        
        for i, ring in enumerate(self.atom_rings):
            ring_set = set(ring)
            if len(ring_set.intersection(self.path_atoms_set)) >= 2:
                self.backbone_expansion_atoms.update(ring_set)
                self.backbone_rings_indices.append(i)
            else:
                self.sidechain_rings_indices.append(i)

        self.structural_backbone_atoms = self.path_atoms_set.union(self.backbone_expansion_atoms)
        self.real_sidechain_atoms = self.all_atom_indices - self.structural_backbone_atoms

    def get_features(self):
        keys = [
            'M_MainChain_Len', 'M_MainChain_AtomFrac', 'M_MainChain_MassFrac',
            'M_SideChain_AtomFrac', 'M_SideChain_MassFrac',
            'M_SideChain_Count', 'M_SideChain_Len_Max', 'M_SideChain_Len_Mean',
            'M_Branching_Density', 'M_Branch_Dist_Mean', 'M_Branch_Dist_Std',
            'M_Ring_Count_BB', 'M_Ring_Count_SC', 'M_Ring_BB_AtomFrac',
            'M_Ring_Aromatic_Frac', 'M_Ring_Aliphatic_Frac', 'M_Ring_Dist_Mean',
            'M_RotBond_BB_Frac', 'M_RotBond_SC_Frac', 'M_Unsaturation_BB_Frac',
            'M_Linker_Degree_Sum', 'M_Linker_Neighbors_Mass',
            'M_Mass_Moment_From_BB', 'M_SC_Center_Of_Mass_Depth',
            'M_BB_Connectivity_Index', 'M_Longest_Path_Ratio', 'M_Wiener_Index_Norm',
            'M_HeteroAtom_BB_Frac', 'M_HeteroAtom_SC_Frac'
        ]

        if not self.valid: return {k: 0.0 for k in keys}
        mol = self.mol
        feats = {}
        
        num_atoms = len(self.all_atom_indices)
        if num_atoms == 0: return {k: 0.0 for k in keys}
        
        len_bb_path = len(self.path_atoms_set)
        num_bb_struct = len(self.structural_backbone_atoms)
        num_sc = len(self.real_sidechain_atoms)
        
        mass_bb = sum([get_atom_mass_with_hs(mol.GetAtomWithIdx(i)) for i in self.structural_backbone_atoms])
        mass_sc = sum([get_atom_mass_with_hs(mol.GetAtomWithIdx(i)) for i in self.real_sidechain_atoms])
        total_mass = mass_bb + mass_sc
        
        feats['M_MainChain_Len'] = len_bb_path
        feats['M_MainChain_AtomFrac'] = num_bb_struct / num_atoms
        feats['M_MainChain_MassFrac'] = mass_bb / total_mass if total_mass > 0 else 0
        feats['M_SideChain_AtomFrac'] = num_sc / num_atoms
        feats['M_SideChain_MassFrac'] = mass_sc / total_mass if total_mass > 0 else 0
        
        sc_lengths = []
        branch_points = 0 
        branch_indices_path = []
        
        if num_sc > 0 and num_bb_struct > 0:
            bb_list = list(self.structural_backbone_atoms)
            for sc_idx in self.real_sidechain_atoms:
                d = np.min(self.dist_matrix[sc_idx, bb_list])
                sc_lengths.append(d)
        
        for idx in self.structural_backbone_atoms:
            atom = mol.GetAtomWithIdx(idx)
            is_branch = False
            for n in atom.GetNeighbors():
                if n.GetIdx() in self.real_sidechain_atoms:
                    is_branch = True
                    break
            if is_branch:
                branch_points += 1
                if idx in self.path_tuple:
                    branch_indices_path.append(self.path_tuple.index(idx))
        
        feats['M_SideChain_Count'] = branch_points
        feats['M_SideChain_Len_Max'] = max(sc_lengths) if sc_lengths else 0
        feats['M_SideChain_Len_Mean'] = np.mean(sc_lengths) if sc_lengths else 0
        feats['M_Branching_Density'] = branch_points / len_bb_path if len_bb_path > 0 else 0
        
        if len(branch_indices_path) >= 2:
            dists = np.diff(sorted(branch_indices_path))
            feats['M_Branch_Dist_Mean'] = np.mean(dists)
            feats['M_Branch_Dist_Std'] = np.std(dists) 
        else:
            feats['M_Branch_Dist_Mean'] = 0; feats['M_Branch_Dist_Std'] = 0

        n_rings = len(self.atom_rings)
        rings_bb = len(self.backbone_rings_indices)
        rings_sc = len(self.sidechain_rings_indices)
        aro_rings = sum(1 for r in self.atom_rings if all(mol.GetAtomWithIdx(x).GetIsAromatic() for x in r))
        ali_rings = n_rings - aro_rings
        atoms_in_rings_flat = set()
        for r in self.atom_rings: atoms_in_rings_flat.update(r)
        bb_atoms_in_ring = len(self.structural_backbone_atoms.intersection(atoms_in_rings_flat))
        
        feats['M_Ring_Count_BB'] = rings_bb
        feats['M_Ring_Count_SC'] = rings_sc
        feats['M_Ring_BB_AtomFrac'] = bb_atoms_in_ring / num_bb_struct if num_bb_struct > 0 else 0
        feats['M_Ring_Aromatic_Frac'] = aro_rings / n_rings if n_rings > 0 else 0
        feats['M_Ring_Aliphatic_Frac'] = ali_rings / n_rings if n_rings > 0 else 0

        r_pos = []
        for r_idx in self.backbone_rings_indices:
            ring = self.atom_rings[r_idx]
            inter = set(ring).intersection(self.path_atoms_set)
            indices = [self.path_tuple.index(x) for x in inter if x in self.path_tuple]
            if indices: r_pos.append(np.mean(indices))
        r_pos.sort()
        feats['M_Ring_Dist_Mean'] = np.mean(np.diff(r_pos)) if len(r_pos) >= 2 else 0
            
        rot_bb_approx = max(0, num_bb_struct - bb_atoms_in_ring)
        feats['M_RotBond_BB_Frac'] = rot_bb_approx / num_bb_struct if num_bb_struct > 0 else 0
        sc_atoms_in_ring = len(self.real_sidechain_atoms.intersection(atoms_in_rings_flat))
        rot_sc_approx = max(0, num_sc - sc_atoms_in_ring)
        feats['M_RotBond_SC_Frac'] = rot_sc_approx / num_sc if num_sc > 0 else 0
        unsat_bb = sum(1 for i in self.structural_backbone_atoms 
                       if str(mol.GetAtomWithIdx(i).GetHybridization()) in ['SP2', 'SP'] 
                       and i not in atoms_in_rings_flat)
        feats['M_Unsaturation_BB_Frac'] = unsat_bb / num_bb_struct if num_bb_struct > 0 else 0

        l_neighs = []
        for d in [mol.GetAtomWithIdx(i) for i in self.dummies]:
            l_neighs.extend(d.GetNeighbors())
        feats['M_Linker_Degree_Sum'] = sum([n.GetDegree() for n in l_neighs])
        
        nb_mass = 0
        visited_l = set(self.dummies)
        for ln in l_neighs:
            nb_mass += get_atom_mass_with_hs(ln); visited_l.add(ln.GetIdx())
            for nn in ln.GetNeighbors():
                if nn.GetIdx() not in visited_l and nn.GetAtomicNum()!=0:
                    nb_mass += get_atom_mass_with_hs(nn); visited_l.add(nn.GetIdx())
        feats['M_Linker_Neighbors_Mass'] = nb_mass

        m_moment, sc_depth_sum = 0.0, 0.0
        if num_sc > 0 and num_bb_struct > 0:
            bb_list = list(self.structural_backbone_atoms)
            for idx in self.real_sidechain_atoms:
                m = get_atom_mass_with_hs(mol.GetAtomWithIdx(idx))
                d = np.min(self.dist_matrix[idx, bb_list])
                m_moment += m * d
                sc_depth_sum += d
        feats['M_Mass_Moment_From_BB'] = m_moment
        feats['M_SC_Center_Of_Mass_Depth'] = sc_depth_sum / num_sc if num_sc > 0 else 0

        bb_degs = [mol.GetAtomWithIdx(i).GetDegree() for i in self.structural_backbone_atoms]
        feats['M_BB_Connectivity_Index'] = np.mean(bb_degs) if bb_degs else 0
        try:
            rw = Chem.RWMol(mol)
            for d in sorted(self.dummies, reverse=True): rw.RemoveAtom(d)
            dm = Chem.GetDistanceMatrix(rw)
            feats['M_Wiener_Index_Norm'] = 0.5 * np.sum(dm) / (num_atoms**2) if num_atoms>0 else 0
            feats['M_Longest_Path_Ratio'] = len_bb_path / np.max(dm) if np.max(dm)>0 else 1.0
        except:
            feats['M_Wiener_Index_Norm'] = 0; feats['M_Longest_Path_Ratio'] = 0

        h_bb = sum(1 for i in self.structural_backbone_atoms if mol.GetAtomWithIdx(i).GetAtomicNum() not in (1,6))
        h_sc = sum(1 for i in self.real_sidechain_atoms if mol.GetAtomWithIdx(i).GetAtomicNum() not in (1,6))
        feats['M_HeteroAtom_BB_Frac'] = h_bb/num_bb_struct if num_bb_struct > 0 else 0
        feats['M_HeteroAtom_SC_Frac'] = h_sc/num_sc if num_sc > 0 else 0
        return feats

# ------------------------------------------------------------------------------
# 2. 核心逻辑：Polymer Fingerprint Generator
# ------------------------------------------------------------------------------
class PolymerFingerprintGenerator:
    def __init__(self):
        self.q_names = [d[0] for d in Descriptors.descList if not d[0].startswith('fr_')]
        self.q_calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.q_names)
        
        raw_blocks = {
            'Benzene': 'c1ccccc1', 
            'Ph_1,4': 'c1cc([*])ccc1[*]',   
            'Ph_1,3': 'c1c([*])cccc1[*]',   
            'Ph_1,2': 'c1c([*])c([*])cccc1',
            'Naphthalene': 'c1cccc2ccccc12',
            'Furan': 'o1cccc1', 
            'Thiophene': 's1cccc1', 
            'Pyridine': 'c1ncccc1',
            'Cyclohexane': 'C1CCCCC1', 
            'Adamantane': 'C1C2CC3CC1CC(C2)C3', 
            'Fluorene': 'C1c2ccccc2-c3ccccc13', 
            'Cyclopropane': 'C1CC1',            
            'Ether': '[OD2]([#6;!$(C=O)])[#6;!$(C=O)]', 
            'Ester': '[#6]C(=O)O[#6]',          
            'Amide': '[#6]C(=O)N[#6]',          
            'Urethane': '[#6]NC(=O)O[#6]',      
            'Carbonate': '[#6]OC(=O)O[#6]',     
            'Imide': 'C(=O)NC(=O)',             
            'Anhydride': 'C(=O)OC(=O)',         
            'Ketone': '[#6]C(=O)[#6;!O;!N]',    
            'Sulfone': 'S(=O)(=O)',             
            'Isopropylidene': 'C(C)(C)',        
            'CF3': 'C(F)(F)F',                  
            'Siloxane': '[Si]O[Si]',            
            'DoubleBond': '[CX3]=[CX3]',        
            'Epoxy': 'C1OC1',                   
        }
        
        self.block_pats = {}
        for k, v in raw_blocks.items():
            if not v: continue
            pat = Chem.MolFromSmarts(v)
            if pat is not None:
                self.block_pats[f"B_{k}"] = pat

    def _make_cyclic_mol(self, mol_in):
        if mol_in is None: return None
        try:
            tm = Chem.RWMol(mol_in)
            ds = [a for a in tm.GetAtoms() if a.GetAtomicNum() == 0]
            if len(ds) != 2: return mol_in 
            
            n0_idx = ds[0].GetNeighbors()[0].GetIdx()
            n1_idx = ds[1].GetNeighbors()[0].GetIdx()
            
            if tm.GetBondBetweenAtoms(n0_idx, n1_idx) is None:
                tm.AddBond(n0_idx, n1_idx, Chem.BondType.SINGLE)
            
            d_indices = sorted([d.GetIdx() for d in ds], reverse=True)
            for idx in d_indices: tm.RemoveAtom(idx)
            
            Chem.SanitizeMol(tm)
            rdDepictor.Compute2DCoords(tm)
            return tm
        except:
            return mol_in

    def _get_triples_details(self, mol):
        triple_map = defaultdict(list)
        if mol is None: return triple_map
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0: continue
            
            ns = [n for n in atom.GetNeighbors() if n.GetAtomicNum() != 0]
            if len(ns) < 2: continue
            
            c_idx = atom.GetIdx()
            c_lbl = f"{atom.GetSymbol()}{atom.GetDegree()}"
            
            for n1, n2 in itertools.combinations(ns, 2):
                l1 = f"{n1.GetSymbol()}{n1.GetDegree()}"
                l2 = f"{n2.GetSymbol()}{n2.GetDegree()}"
                
                labels = sorted([l1, l2])
                triple_str = f"{labels[0]}-{c_lbl}-{labels[1]}"
                
                triple_map[triple_str].append((c_idx, n1.GetIdx(), n2.GetIdx()))
        return triple_map

# ------------------------------------------------------------------------------
# 3. Streamlit 交互与可视化
# ------------------------------------------------------------------------------

st.title("🧬 Polymer Fingerprint Explorer (Pro)")
st.markdown("""
### The ABQM Fingerprint System
*   **M (Morphology)**: **Topological Architecture** — *Backbone/Side-chain decomposition & mass distribution moments.*
*   **B (Blocks)**: **Substructural Motifs** — *Detection of specific functional moieties and chemical building blocks.*
*   **A (Atomic)**: **Local Micro-Environments** — *Atom-centered connectivity triples & neighborhood analysis.*
*   **Q (Quantitative)**: **Physicochemical Descriptors** — *Full set of scalar molecular properties.*
""")

# --- Sidebar ---
st.sidebar.header("Configuration")
default_smiles = "[*]C(=O)c1ccc(C(=O)OCC(C)O[*])cc1" 
smiles_input = st.sidebar.text_area("Monomer SMILES", value=default_smiles, height=100)

process = True
if st.sidebar.button("Analyze Structure", type="primary"):
    process = True

gen = PolymerFingerprintGenerator()

if process and smiles_input:
    mol_linear = Chem.MolFromSmiles(smiles_input)
    if not mol_linear:
        st.error("Invalid SMILES string.")
        st.stop()

    # Calculation
    morph_engine = MorphologicalEngine(mol_linear)
    mol_cyclic = gen._make_cyclic_mol(mol_linear)
    
    if not mol_cyclic:
        st.error("Cyclization failed. Please check dummy atoms [*].")
        st.stop()

    # ==========================================================================
    # 数据导出逻辑 (Full CSV)
    # ==========================================================================
    # 1. M Data
    full_data = {}
    if morph_engine.valid:
        full_data.update(morph_engine.get_features())
    
    # 2. B Data (Include 0 counts)
    for name, pat in gen.block_pats.items():
        count = len(mol_cyclic.GetSubstructMatches(pat))
        full_data[f"B_{name}"] = count
    
    # 3. A Data (Only detected ones)
    triple_details = gen._get_triples_details(mol_cyclic)
    for t, instances in triple_details.items():
        full_data[f"A_{t}"] = len(instances)
        
    # 4. Q Data
    all_desc = gen.q_calc.CalcDescriptors(mol_cyclic)
    for name, val in zip(gen.q_names, all_desc):
        # Clean data
        if not np.isfinite(val): val = 0
        full_data[f"Q_{name}"] = val
    
    df_export = pd.DataFrame([full_data])
    csv_data = df_export.to_csv(index=False).encode('utf-8')
    
    st.sidebar.markdown("---")
    st.sidebar.download_button(
        label="📥 Download Full Fingerprint CSV",
        data=csv_data,
        file_name="ABQM_Full_Fingerprint.csv",
        mime="text/csv",
        help="Contains M (29), B (Defined), A (Detected), and Q (208) features."
    )

    # ==========================================================================
    # UI Visualization
    # ==========================================================================
    tab_m, tab_b, tab_a, tab_q = st.tabs(["🔹 M: Morphology", "🔹 B: Blocks", "🔹 A: Atomics", "🔹 Q: Properties"])

    # TAB 1: M-Fingerprint (Linear)
    with tab_m:
        st.subheader("Morphology: Backbone vs. Sidechain")
        col1, col2 = st.columns([3, 2])
        with col1:
            if morph_engine.valid:
                bb_atoms = list(morph_engine.structural_backbone_atoms)
                sc_atoms = list(morph_engine.real_sidechain_atoms)
                
                hl_colors = {}
                for idx in bb_atoms: hl_colors[idx] = STYLE_COLS["backbone"]
                for idx in sc_atoms: hl_colors[idx] = STYLE_COLS["sidechain"]
                
                svg_m = draw_mol_svg(mol_linear, highlight_atoms=bb_atoms+sc_atoms, highlight_colors=hl_colors)
                st.image(svg_m, use_container_width=True)
                
                st.markdown(f"""
                <div style='display: flex; gap: 10px;'>
                    <span style='color: rgb{tuple(int(c*255) for c in STYLE_COLS["backbone"])}; font-weight: bold;'>● Backbone</span>
                    <span style='color: rgb{tuple(int(c*255) for c in STYLE_COLS["sidechain"])}; font-weight: bold;'>● Sidechain</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Needs two [*] dummy atoms.")

        with col2:
            st.markdown("#### Morphological Metrics")
            if morph_engine.valid:
                m_feats = morph_engine.get_features()
                groups = {
                    "A. Size & Mass": ['M_MainChain_Len', 'M_MainChain_AtomFrac', 'M_MainChain_MassFrac', 'M_SideChain_AtomFrac', 'M_SideChain_MassFrac'],
                    "B. Topology": ['M_SideChain_Count', 'M_SideChain_Len_Max', 'M_SideChain_Len_Mean', 'M_Branching_Density', 'M_Branch_Dist_Mean', 'M_Branch_Dist_Std'],
                    "C. Rings": ['M_Ring_Count_BB', 'M_Ring_Count_SC', 'M_Ring_BB_AtomFrac', 'M_Ring_Aromatic_Frac', 'M_Ring_Aliphatic_Frac', 'M_Ring_Dist_Mean'],
                    "D. Flexibility": ['M_RotBond_BB_Frac', 'M_RotBond_SC_Frac', 'M_Unsaturation_BB_Frac'],
                    "E. Linker/Mass": ['M_Linker_Degree_Sum', 'M_Linker_Neighbors_Mass', 'M_Mass_Moment_From_BB', 'M_SC_Center_Of_Mass_Depth'],
                    "F. Complexity": ['M_BB_Connectivity_Index', 'M_Longest_Path_Ratio', 'M_Wiener_Index_Norm', 'M_HeteroAtom_BB_Frac', 'M_HeteroAtom_SC_Frac']
                }
                flat_data = []
                for group, keys in groups.items():
                    for k in keys:
                        val = m_feats.get(k, 0)
                        flat_data.append({"Category": group, "Feature": k, "Value": f"{val:.4f}"})
                st.dataframe(pd.DataFrame(flat_data), height=500, hide_index=True, use_container_width=True)

    # TAB 2: B-Fingerprint (Cyclic)
    with tab_b:
        st.subheader("Substructure Detection (Cyclized)")
        
        found_blocks = {}
        for name, pat in gen.block_pats.items():
            matches = mol_cyclic.GetSubstructMatches(pat)
            if matches: found_blocks[name] = matches

        if not found_blocks:
            st.info("No blocks detected.")
        else:
            col_b1, col_b2 = st.columns([1, 2])
            with col_b1:
                st.markdown("**Detected Blocks**")
                selection = st.radio("Select to Highlight:", sorted(list(found_blocks.keys())))
                st.metric(label="Occurrences", value=len(found_blocks[selection]))

            with col_b2:
                match_indices_tuple = found_blocks[selection]
                flat_indices = []
                for t in match_indices_tuple: flat_indices.extend(t)
                
                hl_dict = {i: STYLE_COLS["highlight_block"] for i in flat_indices}
                svg_b = draw_mol_svg(mol_cyclic, highlight_atoms=flat_indices, highlight_colors=hl_dict)
                st.image(svg_b, use_container_width=True)

    # TAB 3: A-Fingerprint (Cyclic)
    with tab_a:
        st.subheader("Atomic Triples Explorer (Cyclized)")
        
        if not triple_details:
            st.warning("Molecule too small.")
        else:
            # 排序：从高频到低频
            sorted_triples = sorted(triple_details.keys(), key=lambda k: len(triple_details[k]), reverse=True)
            
            # 使用与 B 层完全相同的布局比例 (1:2)
            col_a1, col_a2 = st.columns([1, 2])
            
            with col_a1:
                st.markdown("**Detected Triples**")
                # 使用 Radio 类似 B 指纹的选择逻辑
                selected_triple = st.radio(
                    "Select to Highlight:", 
                    options=sorted_triples,
                    format_func=lambda x: f"{x}"
                )
                
                count_a = len(triple_details[selected_triple])
                st.metric(label="Occurrences", value=count_a)
                
                st.markdown(f"""
                <div style='margin-top:20px; font-size:0.9em; border-top:1px solid #ddd; padding-top:10px;'>
                    <div><span style='color:rgb{tuple(int(c*255) for c in STYLE_COLS["focus"])}; font-weight:bold'>● Center Atom</span></div>
                    <div><span style='color:rgb{tuple(int(c*255) for c in STYLE_COLS["context"])}; font-weight:bold'>● Neighbor Atoms</span></div>
                </div>
                """, unsafe_allow_html=True)

            with col_a2:
                instances = triple_details[selected_triple]
                hl_colors_a = {}
                highlight_indices = []
                
                for (c_idx, n1_idx, n2_idx) in instances:
                    hl_colors_a[c_idx] = STYLE_COLS["focus"]
                    if n1_idx not in hl_colors_a: hl_colors_a[n1_idx] = STYLE_COLS["context"]
                    if n2_idx not in hl_colors_a: hl_colors_a[n2_idx] = STYLE_COLS["context"]
                    highlight_indices.extend([c_idx, n1_idx, n2_idx])
                
                svg_a = draw_mol_svg(mol_cyclic, highlight_atoms=list(set(highlight_indices)), highlight_colors=hl_colors_a)
                st.image(svg_a, use_container_width=True)

    # TAB 4: Q-Fingerprint (Full Table)
    with tab_q:
        st.subheader("Full Physicochemical Descriptor Table")
        
        if mol_cyclic:
            df_q = pd.DataFrame({
                "Descriptor": gen.q_names,
                "Value": all_desc # 已经在导出逻辑中算过了，但这里为了UI显示需要list格式
            })
            df_q['Value'] = df_q['Value'].fillna(0).replace([np.inf, -np.inf], 0)
            
            st.dataframe(
                df_q, 
                use_container_width=True, 
                height=600,
                column_config={
                    "Descriptor": st.column_config.TextColumn("Descriptor Name", help="RDKit Descriptor"),
                    "Value": st.column_config.NumberColumn("Calculated Value", format="%.4f")
                }
            )