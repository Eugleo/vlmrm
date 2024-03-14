import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from multiclass_evaluator import _load_videos

if __name__ == "__main__":
    results_filename, video_filename = "data/evaluation/results/video_test_results.csv", "/data/datasets/habitat_recordings/2024-03-06/bedroom/bedroom_4.mp4"
    # filename = "out/Exp_20240313_202450/results.csv"
    # filename = "out/Exp_20240313_202832/results.csv"

    # video_filename = "/data/datasets/habitat_recordings/2024-03-12/living_room.mp4"

    labels_to_print = ["living_room", "kitchen"]

    df = pd.read_csv(results_filename)

    probabilities = df["probability"]
    labels = df["label"]
    # convert probabilties from strings formatted like `tensor(0.0769, device='cuda:0')` to floats
    probabilities = probabilities.apply(lambda x: float(x.split(",")[0].split("(")[1]))
    probs_with_labels = {label: [] for label in set(labels)}
    for label, prob in zip(labels, probabilities):
        probs_with_labels[label].append(prob)

    smoothing_window = 1
    smoothed_probs_with_labels = {label: pd.Series(probs).rolling(smoothing_window).mean() for label, probs in probs_with_labels.items()}   
    # for label, probs in smoothed_probs_with_labels.items():
    #     plt.plot(probs, label=label)

    # plot the probabilities over time, grouping those with the same label and plotting each label with a different color
    for label, probs in probs_with_labels.items():
        if label not in labels_to_print:
            continue
        plt.plot(probs, label=label)
        

    # add the legend
    # plt.legend()
    # save the plot to plot.png
    # plt.savefig("plot.png")

    video = _load_videos([video_filename])[0]
    print("frames in video", len(video))
    print("num probabilities", len(list(smoothed_probs_with_labels.values())[0]))

    plot_height_px = 100

    # now create a new video with the probabilities for each frame of the video overlaid on the frame
    out_file = "video_with_probs.mp4"
    # create a VideoWriter object
    out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"mp4v"), 30, (video[0].shape[1], video[0].shape[0] + plot_height_px))
    # out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*"mp4v"), 30, (video[0].shape[1], video[0].shape[0]))

    

    for i, frame in enumerate(video[:30]):
        # get the probabilities for this frame
        frame_probs = [(label, probs[i]) for label, probs in smoothed_probs_with_labels.items()]
        
        # convert the tensor frame to a numpy array
        frame_np = frame.cpu().numpy()

        # convert bgr to rgb
        # frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        
        # write the probabilities on the frame
        # for label, prob in frame_probs:
        #     cv2.putText(frame_np, f"{label}: {prob:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # create the matplotlib plot
        display_window = 10
        # for label, probs in smoothed_probs_with_labels.items():
        #     plt.plot(probs[max(0, i-display_window):i+display_window], label=label)
        # set the x-axis to be centered around the current frame
        plt.xlim(i-display_window, i+display_window)
        
        # plt.legend()

        # make the plot wide and short
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        plt.gcf().set_size_inches(frame_np.shape[1] * 0.95 * px, plot_height_px*px)
        # display the legend to the side of the plot
        plt.subplots_adjust(right=0.7)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.savefig("plot.png")

        # write the matplotlib plot from above on the frame
        plot = cv2.imread("plot.png")


        # scale plot down by 2
        # plot = cv2.resize(plot, (plot.shape[1] // 2, plot.shape[0] // 2))
        
        # scale frame up by the greatest integer factor so that it's under 1920x1080
        # scale_factor = int(min(1920 / frame_np.shape[1], 1080 / frame_np.shape[0]))
        # scale_factor = 2
        # frame_np = cv2.resize(frame_np, (frame_np.shape[1] * scale_factor, frame_np.shape[0] * scale_factor))
        
        # print("plot shape", plot.shape)
        # print("frame shape before extension", frame_np.shape)

        # extend the frame downward to fit the plot by adding 255s to the bottom
        frame_np = np.concatenate([frame_np, np.ones((plot.shape[0], frame_np.shape[1], 3), dtype=np.uint8) * 255], axis=0)

        frame_np[-plot.shape[0]-1:-1, ((frame_np.shape[1] - plot.shape[1]) // 2):((frame_np.shape[1] - plot.shape[1]) // 2) + plot.shape[1]] = plot
        
        # check that the frame shape matches the shape out expects
        # print("frame shape after extension", frame_np.shape)
        # print("expected shape", (video[0].shape[1], video[0].shape[0] + plot_height_px))
        
        # frame_np[0:plot.shape[0], 0:plot.shape[1]] = plot

        # write the frame to the output video
        out.write(frame_np)

    # release the VideoWriter object
    out.release()
    print(f"video with probabilities written to {out_file}")
