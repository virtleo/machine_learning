def agree(h,e,elabel):
    ret=all(h[i]==e[i] or h[i]=='?' for i in range(len(h)))
    return ret if elabel else not ret
def find_s(examples):
    h=['$']*len(attributes)
    for e in examples:
        if not e[1]:
            continue
        if not agree(h,e[0],e[1]):
            h=[e[0][i] if h[i]=='$' else (h[i] if e[0][i]==h[i] else '?') for i in range(len(h))]
    return h
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
print(f"【添加新的训练样例后：】")
print("【find_s算法输出：】")
print(find_s(examples))
