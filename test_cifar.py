import tonic

from datasets.cifar import CIFAR10DVSLDM, CIFAR10DVSWrapper
import matplotlib.pyplot as plt

# dataset = CIFAR10DVSWrapper(save_to='/raid/home/michael.siegl/projects/SE-adlif/data', ignore_first_timesteps=5)

# sample = dataset[0]
# events, target, block_idx = sample
# print(f"Events shape: {events.shape}")
# print(f"Target: {target}")
# print(f"Block idx shape: {block_idx.shape}")

# # so we need to downsample and bin temporally and then we can transform to frame
# downsampling_factor = 48/128 # along x and y dim
# # each cifar sample is a ~1.3s of recording, so we use a window size of 130ms to get around 10 frames per sample
# # each cifar image is 128x128 so we downsample to 48x48

# # padding, since we want to use a fixed window size of 130ms to get 10 windows for 1.3s of recording we need to pad up to that 


# downsampled = tonic.transforms.Downsample(spatial_factor=downsampling_factor, time_factor=1e-3)(events)
# print(f"Downsampled events shape: {downsampled.shape}")
# print(f"Downsampled events sample: {downsampled[1]}")
# frame = tonic.transforms.ToFrame(sensor_size=(48, 48, 2), time_window=129)(downsampled)
# print(f"Frame shape: {frame.shape}")


# # plot the first frame, or view it as an image
# plt.imshow(frame[0, 0, :, :], cmap='gray')
# plt.title(f"Frame 0, Polarity 0, Target: {target}")
# # save the plot
# plt.savefig(f"frame_0_target_{target}.png")



# for i in range(10):
#     events, target, block_idx = dataset[i]
#     print(f"Sample {i}:")
#     print(f"  Events shape: {events.shape}")
#     print(f"  Target: {target}")
#     print(f"  Block idx shape: {block_idx.shape}")
#     print(f"  Event {1}: {events[1]}, Block idx {1}: {block_idx[1]}")

dataset = CIFAR10DVSLDM(data_path='/raid/home/michael.siegl/projects/SE-adlif/data', ignore_first_timesteps=5)
dataset.setup(None)

train_dataloader = dataset.train_dataloader()

for i, batch in enumerate(train_dataloader):
    events, targets, block_idx = batch
    print(f"Batch {i}:")
    print(f"  Events shape: {events.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Block idx shape: {block_idx.shape}")
    print(f"  Event {1}: {events[1]}, Target {1}: {targets[1]}, Block idx {1}: {block_idx[1]}")
    if i >= 0:  # Just check the first batch
        break
