import wandb
import wandb.errors
import wandb.sdk

# need to check if logged in
login = wandb.login()

class WeightsAndBiasHandler:

    def __init__(self,**kwargs) -> None:
        """
            Args:
                **kwargs () : All kwargs for wandb.init, to initialise the run 

            Returns:
                None
        
        """
        # check if wandb is logged in
        if wandb.login(verify=True):
            ...

            wandb.errors.AuthenticationError

        self.run: wandb.sdk.wandb_run.Run =  wandb.init(
            **kwargs
        )
        pass

    def log(self, data, step):
        """
            Log dict to wandb
        """
        wandb.log(
            data = data,
            step = step
        )
        ...
    
    def log_artifact(self):
        """
            Log artifact to wandb
        """

        wandb.log_artifact(

        )

        wandb.Artifact()
        ...


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