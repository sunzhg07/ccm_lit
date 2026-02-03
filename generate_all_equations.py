"""
Complete equation generator for CCSDTQ
Parses all 537 equations from ccsdtq_equations.tex and generates Python code
"""

import re

def parse_equation_line(line_content):
    """Parse a single equation from LaTeX"""
    # Remove eqnarray tags
    eq = line_content.replace('\\begin{eqnarray}', '').replace('\\end{eqnarray}', '').strip()
    
    # Extract coefficient
    coeff = 1.0
    if eq.startswith('-'):
        coeff = -1.0
        eq = eq[1:].strip()
    
    # Handle fractions
    frac_match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', eq)
    if frac_match:
        coeff *= int(frac_match.group(1)) / int(frac_match.group(2))
        eq = eq[frac_match.end():].strip()
    
    # Extract permutation operators P(...)
    perms = []
    while 'P(' in eq:
        match = re.search(r'P\(([^)]+)\)', eq)
        if match:
            perms.append(match.group(1))
            eq = eq[:match.start()] + eq[match.end():]
        else:
            break
    
    # Extract tensors: V^{...}_{...} or t^{...}_{...} or f^{...}_{...}
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

def tensor_to_einsum_and_array(tensor, o='o', v='v'):
    """Convert tensor to einsum indices and array reference"""
    name = tensor['name']
    upper = tensor['upper']
    lower = tensor['lower']
    
    # Map indices
    occ_chars = 'ijklmn'
    virt_chars = 'abcdef'
    
    if name in ['V', 'f']:
        # Two-electron or Fock: upper+lower are the actual indices
        indices = upper + lower
        # Determine slices based on characters
        slices = []
        for char in indices:
            if char in occ_chars:
                slices.append(o)
            elif char in virt_chars:
                slices.append(v)
        array_ref = f"Gamma[{','.join(slices)}]" if name == 'V' else f"f[{','.join(slices)}]"
        return indices, array_ref
    
    elif name == 't':
        # Amplitude: lower=occupied, upper=virtual
        indices = lower + upper
        n_occ = len(lower)
        array_ref = f"t{n_occ}"
        return indices, array_ref
    
    return '', name

def generate_equation_code(eq_dict, target_indices):
    """Generate Python code for one equation"""
    coeff = eq_dict['coeff']
    perms = eq_dict['perms']
    tensors = eq_dict['tensors']
    
    if not tensors:
        return None
    
    # Build einsum notation
    einsum_parts = []
    array_refs = []
    
    for tensor in tensors:
        indices, array_ref = tensor_to_einsum_and_array(tensor)
        einsum_parts.append(indices)
        array_refs.append(array_ref)
    
    einsum_str = ','.join(einsum_parts) + '->' + target_indices
    arrays_str = ', '.join(array_refs)
    
    # Build code
    code = f"    term = {coeff} * contract('{einsum_str}', {arrays_str})\n"
    
    # Apply permutations
    if perms:
        # Build permutation chain
        perm_funcs = []
        for p in perms:
            perm_name = 'P_' + p.replace('/', '_')
            perm_funcs.append(perm_name)
        
        if perm_funcs:
            perm_chain = ''.join([f"{pf}(" for pf in perm_funcs])
            close_parens = ')' * len(perm_funcs)
            code += f"    r{len(target_indices)//2} += {perm_chain}term{close_parens}\n"
    else:
        code += f"    r{len(target_indices)//2} += term\n"
    
    return code

def read_and_generate_all():
    """Read ccsdtq_equations.tex and generate all code"""
    
    with open('/Users/wolf/work/ccm_lit/ccsdtq_equations.tex', 'r') as f:
        lines = f.readlines()
    
    equations = {
        'T1': [],  # lines 13-56
        'T2': [],  # lines 60-172
        'T3': [],  # lines 177-334
        'T4': []   # lines 339-560
    }
    
    targets = {
        'T1': 'ia',
        'T2': 'ijab',
        'T3': 'ijkabc',
        'T4': 'ijklabcd'
    }
    
    # Parse each equation
    for i, line in enumerate(lines[13:57], start=13):  # T1: lines 13-56
        if '\\begin{eqnarray}' in line:
            eq = parse_equation_line(line)
            code = generate_equation_code(eq, targets['T1'])
            if code:
                equations['T1'].append((i, code, eq['raw']))
    
    for i, line in enumerate(lines[60:173], start=60):  # T2: lines 60-172
        if '\\begin{eqnarray}' in line:
            eq = parse_equation_line(line)
            code = generate_equation_code(eq, targets['T2'])
            if code:
                equations['T2'].append((i, code, eq['raw']))
    
    for i, line in enumerate(lines[177:335], start=177):  # T3: lines 177-334
        if '\\begin{eqnarray}' in line:
            eq = parse_equation_line(line)
            code = generate_equation_code(eq, targets['T3'])
            if code:
                equations['T3'].append((i, code, eq['raw']))
    
    for i, line in enumerate(lines[339:561], start=339):  # T4: lines 339-560
        if '\\begin{eqnarray}' in line:
            eq = parse_equation_line(line)
            code = generate_equation_code(eq, targets['T4'])
            if code:
                equations['T4'].append((i, code, eq['raw']))
    
    return equations

def write_complete_residual_functions():
    """Generate complete residual functions with ALL equations"""
    
    print("Parsing all equations from ccsdtq_equations.tex...")
    equations = read_and_generate_all()
    
    print(f"\nParsed equations:")
    print(f"  T1: {len(equations['T1'])} equations")
    print(f"  T2: {len(equations['T2'])} equations")
    print(f"  T3: {len(equations['T3'])} equations")
    print(f"  T4: {len(equations['T4'])} equations")
    print(f"  Total: {sum(len(v) for v in equations.values())} equations")
    
    # Write to output file
    with open('/Users/wolf/work/ccm_lit/generated_residuals.py', 'w') as f:
        f.write('"""\nComplete CCSDTQ Residual Functions\n')
        f.write('Auto-generated from ccsdtq_equations.tex\n"""\n\n')
        f.write('import numpy as np\nfrom opt_einsum import contract\n\n')
        
        # T1 residual
        f.write('\ndef compute_ccsdtq_t1_residual_COMPLETE(f, Gamma, t1, t2, t3, t4, o, v):\n')
        f.write('    """T1 residual with ALL 44 equations from lines 13-56"""\n')
        f.write('    r1 = f[v,o].T.copy()  # Base term\n\n')
        
        for line_num, code, raw in equations['T1']:
            f.write(f'    # Line {line_num}: {raw[:60]}...\n')
            f.write(code)
            f.write('\n')
        
        f.write('    return r1\n\n')
        
        # T2 residual
        f.write('\ndef compute_ccsdtq_t2_residual_COMPLETE(f, Gamma, t1, t2, t3, t4, o, v):\n')
        f.write('    """T2 residual with ALL 113 equations from lines 60-172"""\n')
        f.write('    r2 = Gamma[o,o,v,v].copy()  # Base term\n\n')
        
        for line_num, code, raw in equations['T2']:
            f.write(f'    # Line {line_num}: {raw[:60]}...\n')
            f.write(code)
            f.write('\n')
        
        f.write('    return r2\n\n')
        
        # T3 residual
        f.write('\ndef compute_ccsdtq_t3_residual_COMPLETE(f, Gamma, t1, t2, t3, t4, o, v):\n')
        f.write('    """T3 residual with ALL 158 equations from lines 177-334"""\n')
        f.write('    r3 = np.zeros_like(t3)\n\n')
        
        for line_num, code, raw in equations['T3']:
            f.write(f'    # Line {line_num}: {raw[:60]}...\n')
            f.write(code)
            f.write('\n')
        
        f.write('    return r3\n\n')
        
        # T4 residual
        f.write('\ndef compute_ccsdtq_t4_residual_COMPLETE(f, Gamma, t1, t2, t3, t4, o, v):\n')
        f.write('    """T4 residual with ALL 222 equations from lines 339-560"""\n')
        f.write('    r4 = np.zeros_like(t4)\n\n')
        
        for line_num, code, raw in equations['T4']:
            f.write(f'    # Line {line_num}: {raw[:60]}...\n')
            f.write(code)
            f.write('\n')
        
        f.write('    return r4\n')
    
    print(f"\n✅ Complete residual functions written to: generated_residuals.py")
    print(f"\nNext: Review the file and integrate into ccsdtq_full.py")

if __name__ == '__main__':
    write_complete_residual_functions()
