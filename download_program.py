import getimages as g

print("Getting objects...")
objects = g.get_all_objects()
print("Objects got")
print("Getting object details...")
image_objects = g.get_object_details(objects)
print("Details got")
print("Filtering objects...")
filtered_objects = g.filter_objects_for_ml()
print("Objects filtered")
print("Downloading images...")
image_info = g.download_images_parallel(filtered_objects)
print(image_info)