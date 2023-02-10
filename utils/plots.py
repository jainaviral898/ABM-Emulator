# import os
# import wandb
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# import torch

# # since we cannot predict for the first lag days 
# pred_len = config.t_steps - config.context_len 

# times = []
# temporal_mse_df = []
# for kdx, i in tqdm(enumerate(range(0, len(X_test), pred_len)), total=len(range(0, len(X_test), pred_len))):
#     Exp_Param = X_test_R0s[kdx]

#     if (os.path.exists(PLOTS_PATH + "/{}_{}".format(args["EXP_TYPE"], Exp_Param))):
#         continue
#     os.makedirs(PLOTS_PATH + "/{}_{}".format(args["EXP_TYPE"], Exp_Param), exist_ok=True)

#     start_time = time.time()

#     input = X_test.squeeze()[i:i+1].copy() # right exclusive means we only take one series
#     # input = np.expand_dims(input, axis=0)

#     preds = []
#     actuals = []
    
    

#     for j in range(0, pred_len):
#         output = model(input)
        
#         squeeze_output = output.numpy().squeeze()
        
#         preds.append(squeeze_output)  
#         actuals.append(y_test[i + j].squeeze())

#         # 1 = Horizon 
#         input = np.roll(input, -1, axis=1)
#         input = np.roll(input, -1, axis=1)
#         input[:, -1, :, :, :args["NUM_STATS"]] = squeeze_output
#         input[:, -1, :, :, args["NUM_STATS"]:] = X_test[i+j, -1, :, :, args["NUM_STATS"]:]

#     preds = np.array(preds)
#     actuals = np.array(actuals)
#     mse = {key: [] for key in args["STATS"]}
#     temporal_mse = {key: [] for key in args["STATS"]}

#     time_taken = time.time() - start_time
#     times.append(time_taken)

#     for sdx, stat in enumerate(args["STATS"]):
#         subplot_rows = 2
#         subplot_cols = pred_len//5
#         fig, ax = plt.subplots(subplot_rows, subplot_cols, figsize=(20, 3))
#         # change to idx = 0 (?)
#         idx = -1
#         for i in range(0, pred_len, 5):
#             ax[0, idx].imshow(actuals[i, :, :, sdx], vmin=0, vmax=1)
#             ax[0, idx].title.set_text("Act_{}".format(i+1))
            
#             ax[1, idx].imshow(preds[i, :, :, sdx], vmin=0, vmax=1)
#             ax[1, idx].title.set_text("Pred_{}".format(i+1))

#             ax[0, idx].axis('off')
#             ax[1, idx].axis('off')
#             idx += 1

#         fig.suptitle("{}_{}_{}".format(args["EXP_TYPE"], Exp_Param, stat))
#         # plt.show()
#         plt.savefig(PLOTS_PATH + "/{}_{}/{}_predictions.png".format(args["EXP_TYPE"], Exp_Param, stat))
#         plt.close()
#         # print("\n")

#         subplot_rows = 3
#         subplot_cols = 3
#         fig, ax = plt.subplots(subplot_rows, subplot_cols, figsize=(5, 7))
#         # idx = 0

#         days = iter(range(9, pred_len, 10))
#         for j in range(0, 3):
#             for k in range(0, 3):
#                 i = next(days)
#                 ax[j, k].imshow(actuals[i, :, :, sdx], vmin=0, vmax=2)
#                 ax[j, k].title.set_text("Act_Day_{}".format(i+1))

#                 ax[j, k].axis('off')

#         fig.suptitle("{}_{}_{}".format(args["EXP_TYPE"], Exp_Param, stat))
#         plt.tight_layout()
#         # plt.show()
#         plt.savefig(PLOTS_PATH + "/{}_{}/{}_actuals.png".format(args["EXP_TYPE"], Exp_Param, stat))
#         plt.close()
#         # print("\n")

#         subplot_rows = 3
#         subplot_cols = 3
#         fig, ax = plt.subplots(subplot_rows, subplot_cols, figsize=(5, 7))
#         # idx = 0

#         days = iter(range(9, pred_len, 10))
#         for j in range(0, 3):
#             for k in range(0, 3):
#                 i = next(days)
#                 ax[j, k].imshow(preds[i, :, :, sdx], vmin=0, vmax=2)
#                 ax[j, k].title.set_text("Pred_Day_{}".format(i+1))

#                 ax[j, k].axis('off')

#         fig.suptitle("{}_{}_{}".format(args["EXP_TYPE"], Exp_Param, stat))
#         plt.tight_layout()
#         # plt.show()
#         plt.tight_layout()
#         plt.savefig(PLOTS_PATH + "/{}_{}/{}_preds.png".format(args["EXP_TYPE"], Exp_Param, stat))
#         plt.close()
#         # print("\n")

#         colors = iter(cm.rainbow(np.linspace(0, 1, len(actuals))))
#         plt.figure(figsize=(5, 5))
#         for udx in range(len(actuals)):
#             plt.tight_layout()
#             plt.scatter(x=actuals[udx, :, :, sdx]*args["BLOCK_COUNT"], y=preds[udx, :, :, sdx]*args["BLOCK_COUNT"], color=next(colors), label='day_'+str(udx))

#         plt.xlabel("Actuals")
#         plt.ylabel("Predictions")
#         fig.suptitle("{}_{}_{}".format(args["EXP_TYPE"], Exp_Param, stat))
#         plt.xlim([0, 1000])
#         plt.ylim([0, 1000])
#         # plt.show()
#         plt.tight_layout()
#         plt.savefig(PLOTS_PATH + "/{}_{}/{}_predictions_vs_actuals_scatter.png".format(args["EXP_TYPE"], Exp_Param, stat))
#         plt.close()
#         # print("\n\n\n")

#         plt.xlabel("days")
#         plt.ylabel(stat)

#         # mse[stat].append(mean_squared_error(preds[:, :, :, sdx], actuals[:, :, :, sdx]))

#         # ensure predictions are cumulative
#         if (sdx == 0):
#             temporal_preds = np.sum(preds, axis=(1, 2))[:, sdx]
#             temporal_preds_rolled = np.roll(temporal_preds, 1)
#             temporal_preds_rolled[0] = 0
#             temporal_preds_rolled = temporal_preds - temporal_preds_rolled 
#             temporal_preds_rolled = np.where(temporal_preds_rolled < 0, 0, temporal_preds_rolled).cumsum()

#             temporal_actuals = np.sum(actuals, axis=(1, 2))[:, sdx]
#             temporal_actuals_rolled = np.roll(temporal_actuals, 1)
#             temporal_actuals_rolled[0] = 0
#             temporal_actuals_rolled = temporal_actuals - temporal_actuals_rolled 
#             temporal_actuals_rolled = np.where(temporal_actuals_rolled < 0, 0, temporal_actuals_rolled).cumsum()

#             plt.plot(temporal_actuals_rolled, label='actual')
#             plt.plot(temporal_preds_rolled, label='prediction')
#             plt.plot(temporal_preds_rolled - temporal_actuals_rolled, label='difference')
#             #temporal_mse[stat].append(mean_squared_error(temporal_preds_rolled, temporal_actuals_rolled))
#             temporal_mse[stat].append(temporal_preds_rolled - temporal_actuals_rolled)
#             if args["CI"] == True:
#                 plt.fill_between(np.arange(0,95), (95*(np.mean(preds, axis=(1, 2)) + 2 * np.std(preds, axis=(1, 2))))[:, sdx], (95*(np.mean(preds, axis=(1, 2)) - 2 * np.std(preds, axis=(1, 2))))[:, sdx], color="red", alpha=0.1, label=f"95% confidence interval")    

#         else:
#             plt.plot(np.sum(actuals, axis=(1, 2))[:, sdx], label='actual')
#             plt.plot(np.sum(preds, axis=(1, 2))[:, sdx], label='prediction')
#             plt.plot(np.sum(preds - actuals, axis=(1, 2))[:, sdx], label='difference')
#             #temporal_mse[stat].append(mean_squared_error(np.sum(preds, axis=(1, 2))[:, sdx], np.sum(actuals, axis=(1, 2))[:, sdx]))
#             temporal_mse[stat].append(np.sum(preds, axis=(1, 2))[:, sdx] - np.sum(actuals, axis=(1, 2))[:, sdx])
#             if args["CI"] == True:
#               plt.fill_between(np.arange(0,95), (95*(np.mean(preds, axis=(1, 2)) + 2 * np.std(preds, axis=(1, 2))))[:, sdx], (95*(np.mean(preds, axis=(1, 2)) - 2 * np.std(preds, axis=(1, 2))))[:, sdx], color="red", alpha=0.1, label=f"95% confidence interval")    


#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(PLOTS_PATH + "/{}_{}/{}_temporal_plot.png".format(args["EXP_TYPE"], Exp_Param, stat))
#         plt.close()

#     with open(args["SAVE_DIR"] + args["EXP_VERSION"] + "/plots_new/{}_{}/".format(args["EXP_TYPE"], Exp_Param) + 'preds.npy', 'wb') as f:
#         np.save(f, preds)

#     with open(args["SAVE_DIR"] + args["EXP_VERSION"] + "/plots_new/{}_{}/".format(args["EXP_TYPE"], Exp_Param) + 'actuals.npy', 'wb') as f:
#         np.save(f, actuals)

#     # with open(args["SAVE_DIR"] + args["EXP_VERSION"] + "/plots_new/{}_{}/".format(args["EXP_TYPE"], Exp_Param) + 'mse.json', 'w') as f:
#     #     json.dump(mse, f)

#     # with open(args["SAVE_DIR"] + args["EXP_VERSION"] + "/plots_new/{}_{}/".format(args["EXP_TYPE"], Exp_Param) + 'temporal_mse.json', 'w') as f:
#     #     json.dump(temporal_mse, f)
    
#     temporal_mse["R0"] = Exp_Param
#     temporal_mse["index"] = kdx
#     temporal_mse_df.append(temporal_mse)
        
# #print("--- per series {:.4f} seconds ---".format(sum(times)/len(times)))