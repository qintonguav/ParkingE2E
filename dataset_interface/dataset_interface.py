from dataset_interface.dataset_real import ParkingDataModuleReal


def get_parking_data(data_mode):
    if data_mode == "real_scene":
        model_class = ParkingDataModuleReal
    else:
         raise ValueError(f"Don't support data_mode '{data_mode}'!")
    return model_class

