from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import os
from tqdm import tqdm # Fancy progress bars

import seaborn as sns

from tensorflow.keras.utils import load_img,img_to_array
from keras.utils import load_img, img_to_array
from keras.applications import xception
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


app = Flask(__name__)

model_1 = pickle.load(open('final_class_model.pkl','rb'))
pre_model=pickle.load(open('final_model.pkl','rb'))

# model_2.make_predict_function()

CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
             'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
Intro=[
	"Black grass, also known as black-grass or blackgrass, typically refers to a problematic weed species that can be found in agricultural fields, especially in Europe. The scientific name for black grass is Alopecurus myosuroides. Here's some information about it:",
	"Charlock, scientifically known as Sinapis arvensis, is another common weed found in agricultural fields and disturbed areas. Here's some information about it:",
	"Cleavers, scientifically known as Galium aparine, is a common annual weed found in various habitats, including agricultural fields, gardens, and waste areas. Here's some information about it:",
	"Common chickweed, scientifically known as Stellaria media, is a widespread annual weed found in gardens, lawns, agricultural fields, and other disturbed areas. Here's some information about it:",
	"Common wheat, scientifically known as Triticum aestivum, is one of the most important cereal crops worldwide, widely cultivated for its grains, which are used for making flour for bread, pasta, pastries, and various other food products. Here's some information about it:",
	"Fat hen, also known as common lambsquarters, scientifically named Chenopodium album, is an annual broadleaf weed that is commonly found in agricultural fields, gardens, waste areas, and disturbed habitats. Here's some information about it:",
	"Loose Silky-bent, also known as Yorkshire Fog or Holcus lanatus, is a perennial grass species commonly found in temperate regions around the world. Here's some information about it:",
	"Maize, also known as corn (scientific name: Zea mays), is one of the most important cereal crops globally and has extensive uses ranging from human consumption to animal feed and industrial applications. Here's some information about it:",
	"Scentless Mayweed, scientifically known as Tripleurospermum inodorum, is an annual herbaceous plant that belongs to the daisy family (Asteraceae). Here's some information about it:",
	"Shepherd's purse, scientifically known as Capsella bursa-pastoris, is a small annual herbaceous plant belonging to the mustard family (Brassicaceae). Here's some information about it:",
	"Small-flowered cranesbill, scientifically known as Geranium pusillum, is a perennial herbaceous plant belonging to the Geraniaceae family. Here's some information about it:",
	"Sugar beet, scientifically known as Beta vulgaris subsp. vulgaris, is a root crop grown primarily for sugar production. Here's some information about it:"]
Reasons=[
	"Black grass is an annual grass weed that thrives in arable land, particularly in cereal crops such as wheat, barley, and oats. It has become a significant problem in many agricultural regions due to its ability to adapt to different conditions and its resistance to herbicides.",
	"Charlock is an annual or winter annual weed that commonly infests arable land, particularly in fields where crops like wheat, barley, and oilseed rape are grown. It thrives in disturbed habitats and can quickly colonize open spaces.",
	"Cleavers typically grow in areas with disturbed soil, such as cultivated fields, gardens, and along roadsides. It thrives in fertile soils and is often found in areas with plenty of sunlight.",
	"Common chickweed thrives in moist, fertile soils with plenty of sunlight but can tolerate a wide range of environmental conditions. It often establishes itself in areas with disturbed soil, such as gardens, fields, and lawns.",
	"Common wheat is grown in a wide range of climates, from temperate to subtropical regions, and it thrives in well-drained, fertile soils with adequate moisture and sunlight. It is an essential staple crop for many countries due to its high nutritional value and versatility in food production.",
	"Fat hen thrives in fertile soils with plenty of sunlight and moisture. It is often found in areas with disturbed soil, such as cultivated fields, gardens, and roadsides. It can quickly colonize open spaces and compete with crops for nutrients, water, and sunlight.",
	"Loose Silky-bent typically grows in a variety of habitats, including grasslands, meadows, pastures, roadsides, and disturbed areas. It prefers moist, fertile soils but can tolerate a wide range of soil types and conditions. Loose Silky-bent is often found in areas with moderate to high rainfall and can establish itself in both sunny and shaded areas.",
	"Maize is grown in various climates worldwide, but it thrives in warm temperatures with adequate moisture during its growing season. It is cultivated for its grains, which are used in various food products, and also for silage, fodder, and biofuel production.",
	"Scentless Mayweed is a common weed found in agricultural fields, gardens, roadsides, and other disturbed habitats. It thrives in a wide range of soil types, including sandy, loamy, and clay soils, and is often found in areas with disturbed soil. It can rapidly colonize open spaces and compete with crops for nutrients, water, and sunlight.",
	"Shepherd's purse is a common weed found in agricultural fields, gardens, lawns, and other disturbed habitats. It thrives in various soil types, including sandy, loamy, and clay soils, and is often found in areas with disturbed soil. It can quickly colonize open spaces and compete with crops for nutrients, water, and sunlight.",
	"Small-flowered cranesbill is a common weed found in gardens, lawns, agricultural fields, and other disturbed habitats. It thrives in a wide range of soil types, including sandy, loamy, and clay soils. It can rapidly colonize open spaces and compete with other plants for nutrients, water, and sunlight.",
	"Sugar beet is cultivated for the sugar content stored in its taproot. It is a major crop in many countries and is grown in temperate regions around the world. Sugar beet is an important source of sugar for both food and industrial purposes."]
Conditions=[
	"Black grass prefers fertile, well-drained soils and thrives in mild, moist climates. It can also tolerate a range of soil types but is commonly found in heavy clay soils.",
	"Charlock prefers fertile, well-drained soils with plenty of sunlight. It can tolerate a range of soil types but is commonly found in cultivated fields, roadsides, and waste areas.",
	"Cleavers prefer moist, fertile soils but can tolerate a range of soil types. They are commonly found in areas where soil has been disturbed, such as tilled fields or areas with low vegetation cover.",
	"Common chickweed is highly adaptable and can grow in various soil types, including sandy, loamy, and clay soils. It prefers cool, moist conditions but can survive in both sunny and shaded areas.",
	"Common wheat grows best in loamy soils with a pH level between 6.0 and 7.5. It requires consistent moisture during its growing season, especially during critical growth stages such as tillering, stem elongation, and grain filling. Wheat is a cool-season crop and performs optimally under temperatures between 15°C and 24°C (59°F to 75°F).",
	"Fat hen is highly adaptable and can grow in a wide range of soil types, including sandy, loamy, and clay soils. It prefers moist, fertile soils but can tolerate drought conditions. Fat hen is commonly found in temperate regions but can also grow in subtropical and tropical climates.",
	"Loose Silky-bent is highly adaptable and can grow in a variety of soil types, including sandy, loamy, and clay soils. It prefers moist conditions but can tolerate drought and waterlogged conditions. Loose Silky-bent is known for its characteristic silky hairs on the leaves and flowering stems, which give it a distinctive appearance.",
	"Maize grows best in well-drained, fertile soils with a pH range of 5.5 to 7.0. It requires a warm growing season with temperatures between 20°C and 30°C (68°F to 86°F) for optimal growth and development. Maize is a sun-loving plant and performs best in areas with full sunlight.",
	"Scentless Mayweed prefers moist, fertile soils but can tolerate drought conditions. It is commonly found in temperate regions but can also grow in subtropical and tropical climates. Scentless Mayweed typically grows in full sun but can also tolerate partial shade.",
	"Shepherd's purse is highly adaptable and can grow in a wide range of environmental conditions. It prefers moist, fertile soils but can tolerate drought conditions. Shepherd's purse typically grows in full sun but can also tolerate partial shade.",
	"Small-flowered cranesbill is adaptable to various environmental conditions. It prefers moist, fertile soils but can tolerate drought conditions. It can grow in both full sun and partial shade.",
	"Sugar beet grows best in well-drained, fertile soils with adequate moisture. It requires a relatively long growing season, typically 90 to 150 days from planting to harvest, and performs well in temperate climates with cool springs and warm summers. Sugar beet is often planted in early spring and harvested in late summer or autumn."]
Impact=[
	"Black grass is harmful to soil and crop production. It competes with crops for nutrients, water, and sunlight, reducing crop yields and quality. Moreover, the presence of black grass can increase weed management costs and contribute to the development of herbicide resistance in other weed species.",
	"Charlock can have both positive and negative impacts on soil health. While its extensive root system can help improve soil structure by aerating the soil and adding organic matter when it decomposes, it can also compete with crops for nutrients, water, and sunlight, reducing crop yields. Additionally, charlock may host pests and diseases that can affect nearby crops.",
	"Cleavers are generally not harmful to soil health. In fact, their extensive root system can help improve soil structure by aerating the soil and adding organic matter when they decompose. However, cleavers can become problematic in agricultural fields if they compete with crops for nutrients, water, and sunlight, potentially reducing crop yields.",
	"Common chickweed is not generally considered harmful to soil health. Its shallow root system can help stabilize soil and prevent erosion, and when it dies back, it contributes organic matter to the soil. However, in agricultural fields, chickweed can compete with crops for nutrients, water, and sunlight, potentially reducing crop yields if not managed effectively.",
	"Wheat cultivation can have both positive and negative impacts on soil health. When managed properly, wheat can improve soil structure and fertility by adding organic matter through crop residues and root biomass. However, intensive monoculture practices and improper soil management can lead to soil erosion, nutrient depletion, and soil degradation over time.",
	"Fat hen can have both positive and negative impacts on soil health. On one hand, its extensive root system can help stabilize soil and prevent erosion. Additionally, when fat hen plants die back, they contribute organic matter to the soil, which can improve soil structure and fertility. However, in agricultural fields, fat hen can become a problem weed, competing with crops for resources and reducing crop yields if not managed effectively.",
	"Loose Silky-bent is not generally considered harmful to soil health. As a grass species, it can help stabilize soil and prevent erosion with its extensive root system. Additionally, when Loose Silky-bent plants die back, they contribute organic matter to the soil, which can improve soil structure and fertility. However, in agricultural settings, Loose Silky-bent can become a problem weed, competing with crops for resources and reducing crop yields if not managed effectively.",
	"Maize cultivation can have both positive and negative impacts on soil health. When managed properly, maize can improve soil structure and fertility by adding organic matter through crop residues and root biomass. Additionally, maize can help control soil erosion with its extensive root system and canopy cover. However, intensive monoculture practices and improper soil management can lead to soil erosion, nutrient depletion, and soil compaction.",
	"Scentless Mayweed is not generally considered harmful to soil health. As a herbaceous plant, it does not have an extensive root system like grasses, but it can help stabilize soil and prevent erosion to some extent. Additionally, when Scentless Mayweed plants decompose, they contribute organic matter to the soil, which can improve soil structure and fertility. However, in agricultural settings, Scentless Mayweed can become a problem weed, competing with crops for resources and reducing crop yields if not managed effectively.",
	"Shepherd's purse is not generally considered harmful to soil health. As a small herbaceous plant, it does not have an extensive root system like grasses, but it can help stabilize soil and prevent erosion to some extent. Additionally, when Shepherd's purse plants decompose, they contribute organic matter to the soil, which can improve soil structure and fertility. However, in agricultural settings, Shepherd's purse can become a problem weed, competing with crops for resources and reducing crop yields if not managed effectively.",
	"Small-flowered cranesbill is not generally considered harmful to soil health. As a perennial herbaceous plant, it does not have an extensive root system like grasses, but it can help stabilize soil and prevent erosion to some extent. Additionally, when small-flowered cranesbill plants decompose, they contribute organic matter to the soil, which can improve soil structure and fertility.",
	"Sugar beet cultivation can have both positive and negative impacts on soil health. As a root crop, sugar beet can help improve soil structure and fertility by breaking up compacted soils and adding organic matter through crop residues. However, intensive cultivation practices and improper soil management can lead to soil erosion, nutrient depletion, and soil compaction."]
Manage=[
	"Effective management of black grass typically involves a combination of cultural, mechanical, and chemical control methods. These may include crop rotation, cultivation techniques to bury seeds, using herbicides (though resistance is a concern), and promoting crop competitiveness through practices like appropriate fertilization and seed rates.",
	"Effective management of charlock typically involves a combination of cultural, mechanical, and chemical control methods. These may include crop rotation, regular cultivation to disturb seedlings, hand weeding, mulching, and the use of herbicides when necessary. Early identification and control measures are crucial to prevent charlock from becoming established and spreading rapidly.",
	"Managing cleavers typically involves a combination of cultural, mechanical, and chemical control methods. These may include crop rotation, regular cultivation to disturb seedlings, hand weeding, mulching, and the use of herbicides when necessary. Early identification and control measures are important to prevent cleavers from becoming established and spreading rapidly.",
	"Managing common chickweed typically involves a combination of cultural, mechanical, and chemical control methods. These may include hand weeding, hoeing, mulching, mowing (in lawns), and the use of herbicides when necessary. Preventing seed production and early intervention are important to control chickweed populations before they become extensive.",
	"Effective management of wheat crops involves various agronomic practices, including crop rotation, proper tillage, balanced fertilization, weed control, disease management, and irrigation management. Crop rotation with legumes or other non-cereal crops can help break pest and disease cycles, improve soil fertility, and reduce weed pressure.",
	"Managing fat hen typically involves a combination of cultural, mechanical, and chemical control methods. These may include crop rotation, hand weeding, hoeing, mulching, mowing (in non-crop areas), and the use of herbicides when necessary. Early intervention and preventing seed production are important to control fat hen populations before they become extensive.",
	"Managing Loose Silky-bent typically involves a combination of cultural, mechanical, and chemical control methods. These may include regular mowing, grazing by livestock, hand weeding, cultivation, and the use of herbicides when necessary. Early intervention and preventing seed production are important to control Loose Silky-bent populations before they become extensive.",
	"Effective management of maize crops involves various agronomic practices, including crop rotation, proper tillage, balanced fertilization, weed control, disease management, and irrigation management. Crop rotation with legumes or other non-cereal crops can help break pest and disease cycles, improve soil fertility, and reduce weed pressure. Additionally, conservation tillage practices, such as no-till or reduced tillage, can help improve soil health and reduce erosion.",
	"Managing Scentless Mayweed typically involves a combination of cultural, mechanical, and chemical control methods. These may include crop rotation, hand weeding, hoeing, mulching, mowing (in non-crop areas), and the use of herbicides when necessary. Early intervention and preventing seed production are important to control Scentless Mayweed populations before they become extensive.",
	"Managing Shepherd's purse typically involves a combination of cultural, mechanical, and chemical control methods. These may include crop rotation, hand weeding, hoeing, mulching, mowing (in non-crop areas), and the use of herbicides when necessary. Early intervention and preventing seed production are important to control Shepherd's purse populations before they become extensive.",
	"Managing small-flowered cranesbill typically involves a combination of cultural, mechanical, and chemical control methods. These may include hand weeding, hoeing, mulching, mowing (in non-crop areas), and the use of herbicides when necessary. Preventing seed production is important to control small-flowered cranesbill populations before they become extensive.",
	"Effective management of sugar beet crops involves various agronomic practices, including crop rotation, proper tillage, balanced fertilization, weed control, disease management, and irrigation management. Crop rotation with non-cereal crops can help break pest and disease cycles, improve soil fertility, and reduce weed pressure. Additionally, conservation tillage practices, such as reduced tillage or no-till, can help improve soil health and reduce erosion."]

NUM_CATEGORIES = len(CATEGORIES)


def predict_label(img_path):
	i = image.load_img(img_path, target_size=(224,224))
	j = img_to_array(i)/255.0
	j = np.expand_dims(j, axis=0) 
	features = pre_model.predict(j)
	q = model_1.predict(features)
	return CATEGORIES[q[0]],Intro[q[0]],Reasons[q[0]],Conditions[q[0]],Impact[q[0]],Manage[q[0]]

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "I’m a passionate web developer with a strong focus on backend development. I enjoy creating efficient, scalable, and secure server-side architectures that power modern web applications. With expertise in server management, APIs, and database design, I specialize in turning complex backend challenges into seamless digital experiences.When I’m not coding, I’m likely exploring new backend technologies, optimizing existing systems, or contributing to projects that make an impact. Let’s build something amazing together!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p,i,r,c,im,m = predict_label(img_path)

	return render_template("index.html", prediction = p, intro=i, reasons=r, conditions=c, impact=im, manage=m , img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)