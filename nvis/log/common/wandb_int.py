# import wandb


# WorkFlow, 
    # login
    # init run
    # offload training stats with regular wandb.log
    # offload numerics stats with wandb.log_artifact


# # need to login
# wandb.login()

# run = wandb.init(
#     # Set the project where this run will be logged
#     project="my-awesome-project",
#     # Track hyperparameters and run metadata
#     config={
#         "learning_rate": 0.01,
#         "epochs": 10,
#     },
# )



# run = wandb.init(
#     # Set the project where this run will be logged
#     project="my-awesome-project",
#     # Track hyperparameters and run metadata
#     config={
#         "learning_rate": lr,
#         "epochs": epochs,
#     },
# )

# offset = random.random() / 5
# print(f"lr: {lr}")

# # simulating a training run
# for epoch in range(2, epochs):
#     acc = 1 - 2**-epoch - random.random() / epoch - offset
#     loss = 2**-epoch + random.random() / epoch + offset
#     print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
#     wandb.log({"accuracy": acc, "loss": loss})


# # Artifacts

# run = wandb.init(project = "artifacts-example", job_type = "add-dataset")
# run.log_artifact(artifact_or_path = "./dataset.h5", name = "my_data", type = "dataset" )