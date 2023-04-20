import pandas
import numpy 
import sys

import imageio
import glob

numpy.set_printoptions(threshold=sys.maxsize)


final_dataset = pandas.DataFrame()

for i in range(10):
  print(i)
  result = []
  for file_path in glob.glob("dataset_raw/" + str(i) + "/*.jpg"):
    # file_path = "dataset_raw/0/img_1.jpg"
    imimage = imageio.v2.imread(file_path)
    imimage = imimage.flatten()
    result.append(imimage)

  result_dataframe = pandas.DataFrame(result)
  result_dataframe = result_dataframe.add_prefix('pixel_')
  result_dataframe['digit'] = i
  final_dataset = pandas.concat([result_dataframe, final_dataset], axis=0)


print(final_dataset)
final_dataset.to_csv("dataset.csv", index=False)

