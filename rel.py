# relationships.csv ilk 5 satırı kontrol et
with open('D:/Huawei Share/KG_RAG_Project/relationships.csv', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < 5:
            print(f"Line {i}: {repr(line)}")
        else:
            break