from preprocessing import preprocess_data
from model_training import train_model
from fairness_analysis import fairness_analysis


def main():
    print("Starting Student Dropout Prediction Pipeline...\n")

    preprocess_data()

    train_model()

    fairness_analysis()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
