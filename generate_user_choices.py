objects = {
    'Coffee Cup': 1,
    'Sponge': 2,
    'Gloves': 1,
    'Plate': 3,
    'PC Fan': 1,
    'Soap': 2,
    'Controller': 1,
    'Cloth': 2,
    'Mouse': 1,
    'Screwdriver': 1,
    'PCB': 1,
    'Cleaning Foam': 2
}

image_per_object = 5
with open('user_choice_0328.csv', 'w') as f:
    for i in range(len(objects)):
        for j in range(image_per_object):
            k = list(objects.keys())[i]
            f.write(f'{i*image_per_object+j+1:02d}.jpg,{k},{objects[k]}\n')