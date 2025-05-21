attributes = [
    ['Sunny', 'Rainy'],
    ['Warm', 'Cold'],
    ['Normal', 'High'],
    ['Strong', 'Light'],
    ['Warm', 'Cool'],
    ['Same', 'Change']
]
examples = [
[['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'], True],
[['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'], True],
[['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'], False],
[['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'], True],
 [['Sunny','Cold','Normal','Strong','Warm','Same'], False]
]
def agree(h,e,elabel):
    ret=all(h[i]==e[i] or h[i]=='?' for i in range(len(h)))
    return ret if elabel else not ret
empty_h=['$']*len(attributes)
full_h=['?']*len(attributes)
def is_special_than(h1, h2, strict=False):
    ret = all(h1[i]==h2[i] or  h2[i]=='?' for i in range(len(h1)))
    return ret
def is_general_than(h1, h2, strict=False):
    ret = all(h1[i]==h2[i] or  h1[i]=='?' for i in range(len(h1)))
    return ret
def min_generalize(h, e):
    ret=[e[i] if (h[i]=='$') else ('?' if h[i]!=e[i] else h[i])for i in range(len(h))]
    return ret
def min_specialize(h,e):
    specs = []
    for i in range(len(h)):
        if h[i] == '?':
            for val in attributes[i]:
                if val != e[i]:
                    new_g = h.copy()
                    new_g[i] = val
                    specs.append(new_g)
    return specs
def generate_SG(examples):
    S,G=[empty_h],[full_h]
    for e in examples:
        if e[1]:  # 正例处理
            G = [g for g in G if agree(g, e[0], True)]
            new_S = []
            for s in S:
                if agree(s, e[0], True):
                    new_S.append(s)
                else:
                    new_S.append(min_generalize(s, e[0]))
            S = []
            for s in new_S:
                if any(is_special_than(s, g) for g in G):
                    S.append(s)
            S = [s for s in S if not any(is_special_than(other, s) for other in S if s != other)]
        else:  # 负例处理
            S = [s for s in S if agree(s, e[0], False)]
            new_G = []
            for g in G:
                if agree(g, e[0], False):
                    new_G.append(g)
                else:
                    specs = min_specialize(g, e[0])
                    valid_specs = [sg for sg in specs if any(is_special_than(s, sg) for s in S)]
                    new_G.extend(valid_specs)
            G = []
            for g in new_G:
                if not any(is_special_than(g, other) for other in new_G if g != other):
                    G.append(g)
    return S, G
def generate_VS(S, G):
    import itertools
    all_hypos = []
    for combo in itertools.product(*[['?']+vals for vals in attributes]):
        h = list(combo)
        if (any(is_special_than(s, h) for s in S) and 
            any(is_special_than(h, g) for g in G)):
            all_hypos.append(h)
    unique_hypos = []
    [unique_hypos.append(h) for h in all_hypos if h not in unique_hypos]
    return sorted(unique_hypos, key=lambda x: x.count('?'), reverse=True)
S,G=generate_SG(examples)
print(f"【添加新的训练样例后：】")
print(f'【最终版本空间】\nS={S}\nG={G}')
VS = generate_VS(S, G)
print(f'【变型空间】')
print(f'S={S}')
for i, h in enumerate(VS):
    print(f"h{i+1}: {h}")
print(f'G={G}')