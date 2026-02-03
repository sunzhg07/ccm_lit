"""
Parser for ccsdtq_equations.tex
Automatically generates Python code for CCSDTQ residual equations
"""

import re
import sys

def parse_latex_equation(latex_str):
    """
    Parse a LaTeX equation string and convert to Python/numpy code
    
    Example input:
        -V^{jk}_{bc}t^{b}_{i}t^{a}_{j}t^{c}_{k}
    
    Returns:
        {
            'coefficient': -1,
            'tensors': [
                {'name': 'V', 'upper': 'jk', 'lower': 'bc'},
                {'name': 't', 'upper': 'b', 'lower': 'i'},
                ...
            ],
            'einsum_str': 'jkbc,bi,aj,ck->ia',
            'permutations': []
        }
    """
    # Remove \begin{eqnarray} and \end{eqnarray}
    latex_str = latex_str.replace('\\begin{eqnarray}', '').replace('\\end{eqnarray}', '').strip()
    
    # Extract coefficient
    coeff = 1.0
    if latex_str.startswith('-'):
        coeff = -1.0
        latex_str = latex_str[1:]
    
    # Check for fractional coefficients
    frac_match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', latex_str)
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        coeff *= num / den
        latex_str = latex_str[frac_match.end():]
    
    # Extract permutation operators
    permutations = []
    perm_pattern = r'P\(([^)]+)\)'
    for match in re.finditer(perm_pattern, latex_str):
        permutations.append(match.group(1))
        latex_str = latex_str.replace(match.group(0), '', 1)
    
    # Extract tensors (V or t with subscripts and superscripts)
    tensors = []
    
    # Pattern for V^{...}_{...} or t^{...}_{...}
    tensor_pattern = r'([Vft])\^?\{([^}]+)\}_?\{([^}]+)\}'
    
    for match in re.finditer(tensor_pattern, latex_str):
        tensor_name = match.group(1)
        upper_indices = match.group(2)
        lower_indices = match.group(3)
        
        tensors.append({
            'name': tensor_name,
            'upper': upper_indices,
            'lower': lower_indices
        })
    
    return {
        'coefficient': coeff,
        'tensors': tensors,
        'permutations': permutations,
        'latex': latex_str
    }


def tensor_to_einsum(tensor_info, target_indices):
    """
    Convert tensor notation to einsum notation
    
    Args:
        tensor_info: dict with 'name', 'upper', 'lower'
        target_indices: final target indices (e.g., 'ia' for T1, 'ijab' for T2)
    
    Returns:
        einsum string component
    """
    name = tensor_info['name']
    
    if name == 'V' or name == 'f':
        # Two-electron or Fock matrix
        # V^{ij}_{ab} -> 'ijab'
        indices = tensor_info['upper'] + tensor_info['lower']
    elif name == 't':
        # Amplitude: t^{a}_{i} or t^{ab}_{ij}
        # Convention: occupied first, then virtual
        indices = tensor_info['lower'] + tensor_info['upper']
    else:
        indices = tensor_info['upper'] + tensor_info['lower']
    
    return indices


def permutation_to_python(perm_str):
    """
    Convert permutation notation to Python function name
    
    Examples:
        'i/j' -> 'P_ij'
        'a/bc' -> 'P_a_bc'
        'ab/cd' -> 'P_ab_cd'
        'a/b/c/d' -> 'P_a_b_c_d'
    """
    # Replace / with _
    return 'P_' + perm_str.replace('/', '_')


def equation_to_code(eq_dict, amplitude_type):
    """
    Generate Python code for a single equation term
    
    Args:
        eq_dict: parsed equation dictionary
        amplitude_type: 'T1', 'T2', 'T3', or 'T4'
    
    Returns:
        Python code string
    """
    coeff = eq_dict['coefficient']
    tensors = eq_dict['tensors']
    perms = eq_dict['permutations']
    
    if not tensors:
        return None
    
    # Determine target indices based on amplitude type
    target_map = {
        'T1': 'ia',
        'T2': 'ijab',
        'T3': 'ijkabc',
        'T4': 'ijklabcd'
    }
    target = target_map[amplitude_type]
    
    # Build einsum string
    einsum_inputs = []
    tensor_names = []
    
    for t_info in tensors:
        einsum_component = tensor_to_einsum(t_info, target)
        einsum_inputs.append(einsum_component)
        
        # Map to actual numpy arrays
        if t_info['name'] == 'V':
            # Determine which V block based on indices
            tensor_names.append(f"Gamma[{map_indices_to_slices(einsum_component)}]")
        elif t_info['name'] == 'f':
            tensor_names.append(f"f[{map_indices_to_slices(einsum_component)}]")
        elif t_info['name'] == 't':
            # Determine t1, t2, t3, or t4
            n_occ = len([c for c in t_info['lower'] if c in 'ijklmn'])
            tensor_names.append(f"t{n_occ}")
        else:
            tensor_names.append(t_info['name'])
    
    # Construct einsum call
    einsum_str = ','.join(einsum_inputs) + '->' + target
    tensor_list = ', '.join(tensor_names)
    
    # Build the code
    code = f"    term = {coeff} * contract('{einsum_str}', {tensor_list}, optimize='optimal')\n"
    
    # Apply permutations
    if perms:
        perm_funcs = [permutation_to_python(p) for p in perms]
        perm_chain = ''.join([f"{pf}(" for pf in perm_funcs])
        closing = ')' * len(perm_funcs)
        code += f"    r{len(target)//2} += {perm_chain}term{closing}\n"
    else:
        code += f"    r{len(target)//2} += term\n"
    
    return code


def map_indices_to_slices(indices):
    """
    Map index string to slices
    'ijab' -> 'o,o,v,v'
    """
    occ_indices = 'ijklmn'
    virt_indices = 'abcdef'
    
    slices = []
    for idx in indices:
        if idx in occ_indices:
            slices.append('o')
        elif idx in virt_indices:
            slices.append('v')
        else:
            slices.append('?')
    
    return ','.join(slices)


def parse_ccsdtq_equations(tex_file):
    """
    Parse the entire ccsdtq_equations.tex file
    """
    with open(tex_file, 'r') as f:
        content = f.read()
    
    # Extract equations by looking for %Equation_extracted@#N patterns
    equation_pattern = r'%Equation_extracted@#(\d+)\s*\\begin\{eqnarray\}(.+?)\\end\{eqnarray\}'
    
    equations = {}
    for match in re.finditer(equation_pattern, content, re.DOTALL):
        eq_num = int(match.group(1))
        eq_latex = match.group(2).strip()
        
        parsed = parse_latex_equation(eq_latex)
        parsed['equation_number'] = eq_num
        equations[eq_num] = parsed
    
    return equations


def generate_residual_functions(equations_dict):
    """
    Generate complete Python residual functions from parsed equations
    """
    # Group by equation ranges
    t1_eqs = {k: v for k, v in equations_dict.items() if 4 <= k <= 18}
    t2_eqs = {k: v for k, v in equations_dict.items() if 19 <= k <= 56}
    t3_eqs = {k: v for k, v in equations_dict.items() if 57 <= k <= 109}
    t4_eqs = {k: v for k, v in equations_dict.items() if 110 <= k <= 183}
    
    print(f"Found {len(t1_eqs)} T1 equations")
    print(f"Found {len(t2_eqs)} T2 equations")
    print(f"Found {len(t3_eqs)} T3 equations")
    print(f"Found {len(t4_eqs)} T4 equations")
    
    # Generate code for each
    for eq_type, eqs, amp_type in [
        ('T1', t1_eqs, 'T1'),
        ('T2', t2_eqs, 'T2'),
        ('T3', t3_eqs, 'T3'),
        ('T4', t4_eqs, 'T4')
    ]:
        print(f"\n{'='*60}")
        print(f"{eq_type} Residual Code:")
        print('='*60)
        
        for eq_num in sorted(eqs.keys()):
            eq_dict = eqs[eq_num]
            code = equation_to_code(eq_dict, amp_type)
            if code:
                print(f"    # Equation {eq_num}: {eq_dict['latex'][:50]}...")
                print(code)


if __name__ == "__main__":
    tex_file = '/Users/wolf/work/ccm_lit/ccsdtq_equations.tex'
    
    print("Parsing CCSDTQ equations from LaTeX file...")
    equations = parse_ccsdtq_equations(tex_file)
    
    print(f"\nTotal equations found: {len(equations)}")
    
    # Display sample
    print("\nSample parsed equations:")
    for eq_num in list(equations.keys())[:5]:
        eq = equations[eq_num]
        print(f"\nEquation {eq_num}:")
        print(f"  Coefficient: {eq['coefficient']}")
        print(f"  Tensors: {eq['tensors']}")
        print(f"  Permutations: {eq['permutations']}")
    
    # Generate code
    print("\n" + "="*70)
    print("GENERATING PYTHON CODE")
    print("="*70)
    generate_residual_functions(equations)
