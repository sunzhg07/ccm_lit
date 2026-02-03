"""
CCDTQ Equation Generator (T1 = 0)
Parses ccsdtq_equations.tex but EXCLUDES all terms involving t1
"""

import re

def parse_equation_line(line_content):
    """Parse a single equation from LaTeX (Same as before)"""
    eq = line_content.replace('\\begin{eqnarray}', '').replace('\\end{eqnarray}', '').strip()
    
    coeff = 1.0
    if eq.startswith('-'):
        coeff = -1.0
        eq = eq[1:].strip()
    
    frac_match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', eq)
    if frac_match:
        coeff *= int(frac_match.group(1)) / int(frac_match.group(2))
        eq = eq[frac_match.end():].strip()
    
    perms = []
    while 'P(' in eq:
        match = re.search(r'P\(([^)]+)\)', eq)
        if match:
            perms.append(match.group(1))
            eq = eq[:match.start()] + eq[match.end():]
        else:
            break
    
    tensors = []
    pattern = r'([Vft])\^?\{([^}]+)\}_?\{([^}]+)\}'
    
    for match in re.finditer(pattern, eq):
        tensor_name = match.group(1)
        upper = match.group(2)
        lower = match.group(3)
        tensors.append({'name': tensor_name, 'upper': upper, 'lower': lower})
    
    return {
        'coeff': coeff,
        'perms': perms,
        'tensors': tensors,
        'raw': line_content
    }

def is_t1_term(tensor):
    """Check if a tensor is t1 (1 upper index, 1 lower index)"""
    if tensor['name'] == 't':
        if len(tensor['upper']) == 1 and len(tensor['lower']) == 1:
            return True
        # Also check for 'fi' or 'fa' in Fock matrix if it implies singles couling? 
        # No, Fock is f_oo, f_vv etc. But f_ov (f^a_i) acts like T1 source.
        # But instructions say "equations that t1 is involved".
        # We will strictly specific terms containing 't' with 1 index.
    return False

def tensor_to_einsum_and_array(tensor, o='o', v='v'):
    """Convert tensor to einsum (Same as before)"""
    name = tensor['name']
    upper = tensor['upper']
    lower = tensor['lower']
    
    occ_chars = 'ijklmn'
    virt_chars = 'abcdef'
    
    if name in ['V', 'f']:
        indices = upper + lower
        slices = []
        for char in indices:
            if char in occ_chars: slices.append(o)
            elif char in virt_chars: slices.append(v)
        
        # Determine array name
        arr = "Gamma" if name == 'V' else "f"
        return indices, f"{arr}[{','.join(slices)}]"
    
    elif name == 't':
        indices = lower + upper
        n_occ = len(lower)
        return indices, f"t{n_occ}"
    
    return '', name

def generate_equation_code(eq_dict, target_indices):
    """Generate code, returning None if T1 is involved"""
    tensors = eq_dict['tensors']
    if not tensors: return None
    
    # FILTER: If ANY tensor is t1, discard the whole term
    for t in tensors:
        if is_t1_term(t):
            return None
    
    # Build code
    einsum_parts = []
    array_refs = []
    
    for tensor in tensors:
        indices, array_ref = tensor_to_einsum_and_array(tensor)
        einsum_parts.append(indices)
        array_refs.append(array_ref)
    
    einsum_str = ','.join(einsum_parts) + '->' + target_indices
    arrays_str = ', '.join(array_refs)
    
    coeff = eq_dict['coeff']
    code = f"    term = {coeff} * contract('{einsum_str}', {arrays_str})\n"
    
    # Permutations
    perms = eq_dict['perms']
    if perms:
        perm_chain = ''.join([f"P_{p.replace('/', '_')}(" for p in perms])
        close_parens = ')' * len(perms)
        code += f"    r{len(target_indices)//2} += {perm_chain}term{close_parens}\n"
    else:
        code += f"    r{len(target_indices)//2} += term\n"
    
    return code

def write_ccdtq_residuals():
    """Generate minimal CCDTQ residuals"""
    print("Generating CCDTQ equations (skipping T1 terms)...")
    
    with open('/Users/wolf/work/ccm_lit/ccsdtq_equations.tex', 'r') as f:
        lines = f.readlines()
        
    targets = {'T2': 'ijab', 'T3': 'ijkabc', 'T4': 'ijklabcd'}
    
    with open('/Users/wolf/work/ccm_lit/ccdtq_residuals.py', 'w') as f:
        f.write('"""\nCCDTQ Residuals (T1 = 0)\nAuto-generated from ccsdtq_equations.tex\n"""\n\n')
        f.write('import numpy as np\nfrom opt_einsum import contract\n')
        f.write('from ccsdtq_permutations import *\n\n')
        
        # T2
        f.write('\ndef compute_ccdtq_t2_residual(f, Gamma, t2, t3, t4, o, v):\n')
        f.write('    """T2 residual (T1 terms removed)"""\n')
        f.write('    r2 = Gamma[o,o,v,v].copy()\n\n')
        count_t2 = 0
        for i, line in enumerate(lines[60:173], start=60):
            if '\\begin{eqnarray}' in line:
                eq = parse_equation_line(line)
                code = generate_equation_code(eq, targets['T2'])
                if code:
                    f.write(f'    # Line {i}\n{code}\n')
                    count_t2 += 1
        f.write('    return r2\n\n')

        # T3
        f.write('\ndef compute_ccdtq_t3_residual(f, Gamma, t2, t3, t4, o, v):\n')
        f.write('    """T3 residual (T1 terms removed)"""\n')
        f.write('    r3 = np.zeros_like(t3)\n\n')
        count_t3 = 0
        for i, line in enumerate(lines[177:335], start=177):
            if '\\begin{eqnarray}' in line:
                eq = parse_equation_line(line)
                code = generate_equation_code(eq, targets['T3'])
                if code:
                    f.write(f'    # Line {i}\n{code}\n')
                    count_t3 += 1
        f.write('    return r3\n\n')

        # T4
        f.write('\ndef compute_ccdtq_t4_residual(f, Gamma, t2, t3, t4, o, v):\n')
        f.write('    """T4 residual (T1 terms removed)"""\n')
        f.write('    r4 = np.zeros_like(t4)\n\n')
        count_t4 = 0
        for i, line in enumerate(lines[339:561], start=339):
            if '\\begin{eqnarray}' in line:
                eq = parse_equation_line(line)
                code = generate_equation_code(eq, targets['T4'])
                if code:
                    f.write(f'    # Line {i}\n{code}\n')
                    count_t4 += 1
        f.write('    return r4\n')
        
    print(f"\nEquation Stats (T1 terms filtered out):")
    print(f"  T2: {count_t2} equations kept")
    print(f"  T3: {count_t3} equations kept")
    print(f"  T4: {count_t4} equations kept")
    print(f"  Saved to ccdtq_residuals.py")

if __name__ == '__main__':
    write_ccdtq_residuals()
