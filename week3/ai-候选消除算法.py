def agree(h, e, elabel):
    """判断假设h是否与样例<e, elabel>一致"""
    if elabel:  # 正例需要严格匹配
        for h_attr, e_attr in zip(h, e):
            if h_attr not in ('?', e_attr):
                return False
        return True
    else:  # 负例不能匹配
        match = True
        for h_attr, e_attr in zip(h, e):
            if h_attr != '?' and h_attr != e_attr:
                match = False
                break
        return not match

def find_S(examples, attributes):
    """Find-S算法实现"""
    h = ['$'] * len(attributes)
    for features, label in examples:
        if not label: continue
        if not agree(h, features, True):
            h = [f if h_attr == '$' else ('?' if h_attr != f else h_attr) 
                for h_attr, f in zip(h, features)]
    return ['?' if x == '$' else x for x in h]

def is_special_than(h1, h2):
    """判断h1是否比h2更特殊"""
    return all(a1 == a2 or a2 == '?' for a1, a2 in zip(h1, h2))

def min_generalize(h, e):
    """正例极小泛化"""
    return [e[i] if h[i] == '$' else ('?' if h[i] != e[i] else h[i]) 
           for i in range(len(h))]

def min_specialize(g, e, attributes):
    """负例极小特化"""
    specs = []
    for i in range(len(g)):
        if g[i] == '?':
            for val in attributes[i]:
                if val != e[i]:
                    new_g = g.copy()
                    new_g[i] = val
                    specs.append(new_g)
    return specs

def generate_SG(examples, attributes):
    """候选消除算法核心实现"""
    S = [['$']*len(attributes)]
    G = [['?']*len(attributes)]
    
    for idx, (features, label) in enumerate(examples):
        print(f"\n=== 处理第{idx+1}个样例 ===")
        print(f"特征：{features}，标签：{label}")
        
        if label:  # 正例处理
            # 剪枝G
            G = [g for g in G if agree(g, features, True)]
            # 泛化S
            new_S = []
            for s in S:
                if agree(s, features, True):
                    new_S.append(s)
                else:
                    new_S.append(min_generalize(s, features))
            # 保留最特殊边界
            S = [s for s in new_S if any(is_special_than(s, g) for g in G)]
            S = [s for s in S if not any(is_special_than(other, s) 
                                      for other in S if s != other)]
        else:  # 负例处理
            # 剪枝S
            S = [s for s in S if agree(s, features, False)]
            # 特化G
            new_G = []
            for g in G:
                if agree(g, features, False):
                    new_G.append(g)
                else:
                    specs = min_specialize(g, features, attributes)
                    new_G.extend([sg for sg in specs 
                                if any(is_special_than(s, sg) for s in S)])
            # 保留最一般边界
            G = [g for g in new_G if not any(is_special_than(g, other) 
                                           for other in new_G if g != other)]
        
        print(f"更新后 S边界：{S}\n更新后 G边界：{G}")
    return S, G

def generate_VS(S, G, attributes):
    """生成图示的变型空间"""
    import itertools
    # 生成所有可能的假设组合（包含通配符）
    all_hypos = []
    for combo in itertools.product(*[['?']+vals for vals in attributes]):
        h = list(combo)
        # 必须介于S和G之间
        if (any(is_special_than(s, h) for s in S) and 
            any(is_special_than(h, g) for g in G)):
            all_hypos.append(h)
    # 去重并按通配符数量排序
    unique_hypos = []
    [unique_hypos.append(h) for h in all_hypos if h not in unique_hypos]
    return sorted(unique_hypos, key=lambda x: x.count('?'), reverse=True)

# --------------------- 测试案例 --------------------
attributes = [
    ['Sunny','Rainy'],
    ['Warm','Cold'],
    ['Normal','High'],
    ['Strong','Light'],
    ['Warm','Cool'],
    ['Same','Change']
]

examples = [
    [['Sunny','Warm','Normal','Strong','Warm','Same'], True],
    [['Sunny','Warm','High','Strong','Warm','Same'], True],
    [['Rainy','Cold','High','Strong','Warm','Change'], False],
    [['Sunny','Warm','High','Strong','Cool','Change'], True]
]

# 执行Find-S算法
print("【Find-S算法输出】")
find_s_result = find_S(examples, attributes)
print(find_s_result)

# 执行候选消除算法
print("\n【候选消除算法过程】")
S, G = generate_SG(examples, attributes)

# 生成变型空间
print("\n【最终版本空间】")
VS = generate_VS(S, G, attributes)
print(f"S边界：{S}")
print(f"G边界：{G}")
print("变型空间假设列表：")
for i, h in enumerate(VS):
    print(f"h{i+1}: {h}")