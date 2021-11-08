from deepforest import main
import matplotlib.pyplot as plt

# initiating model
model = main.deepforest()
model.use_release()

img = model.predict_image(path="./Data/odm_orthophoto.tif", return_plot=True, thickness=10, color=(0, 0, 255))


#predict_image returns plot in BlueGreenRed (opencv style), but matplotlib likes RedGreenBlue, switch the channel order.
plt.imshow(img[:,:,::-1])
plt.imsave("./figures/first_trial.png", img[:,:,::-1])