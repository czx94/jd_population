import json

if __name__ == '__main__':
    with open("loss_table.json", 'r') as load_f:
        load_dict = json.load(load_f)

    loss_table = sorted(load_dict.items(), key=lambda item: item[1])
    print(loss_table[:3])