def agree(h, e, elabel):#判断假设 h 是否与样例 <e, elabel> 一致
    if elabel:#正例
        for h_attr, e_attr in zip(h, e):#zip(h, e) 将列表 h 和 e 对应位置的元素打包成元组
            if h_attr != '?' and h_attr != e_attr:
                return False
        return True
    else:#负例
        for h_attr, e_attr in zip(h, e):
            if h_attr == '?' or h_attr == e_attr:
                return False  # h覆盖了负例，不一致
        return True  # 所有属性都不匹配，一致

def find_S(examples, attributes):
    """
    实现Find-S算法，寻找最特殊假设
    :param examples: 训练样例列表（格式：[[特征], 标签]）
    :param attributes: 属性取值列表（用于初始化假设长度）
    :return: 最特殊假设
    """
    # 初始化最特殊假设：每个属性用'$'表示（比具体值更特殊）
    h = ['$'] * len(attributes)

    for e in examples:
        features, label = e
        if not label:  # 跳过负例
            continue
        # 如果当前假设不覆盖正例
        if not agree(h, features, True):
            for i in range(len(h)):
                if h[i] == '$':
                    h[i] = features[i]
                elif h[i] != features[i]:
                    h[i] = '?'

    # 最终将'$'替换为'?'（因为'$'仅用于初始化，实际有效符号为具体值和'?'）
    h = ['?' if attr == '$' else attr for attr in h]
    return h

# 测试Find-S算法
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
    [['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change'], True]
]

find_s_result = find_S(examples, attributes)
print("Find-S输出:", find_s_result)  # 预期输出: ['Sunny', 'Warm', '?', 'Strong', '?', '?']
    