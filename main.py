from torchvision.models import resnet18
from vector_indexing_database import init_db, insert_vector_and_id, find_vector, dump_db, restore_db_from_file
from utils import extract_features
from PIL import Image

model = resnet18(pretrained=True)

# mark layer to extract features
feature_extraction_layer = model.fc


indexed_database = init_db()

image = Image.open('test_images/1.png')

extracted_feature = extract_features(model, feature_extraction_layer, image)

# add vector in database
insert_vector_and_id(50, extracted_feature, indexed_database)
insert_vector_and_id(30, extracted_feature, indexed_database)

dump_db(indexed_database, 'test')

restored_db = restore_db_from_file('test')
result = find_vector(extracted_feature, restored_db, n_results=5)
print(result)
