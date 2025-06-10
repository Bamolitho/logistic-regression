from regularized_logistic_regression import *
import os

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def wait_for_exit():
    """
    Wait for user to press 'q' or 'enter' to quit
    """
    print("\nPress [Enter] to make another action, or type 'q' to quit.")
    choice = input("Your choice: ").strip().lower()
    return choice != 'q'

def main():
    # Loop for multiple actions
    continue_running = True
    while continue_running: 
        print("1. Plot the decision boundary of the model")
        print("2. Compute accuracy on the training set")
        print("3. Predict")

        choice = int(input("What do you want to do? Choose an integer between 1 and 3 : "))

        if choice == 0:
            return -1
        elif choice == 1 :
            clear_console()
            descision_boundary()
        elif choice == 2 :
            clear_console()
            compute_accuracy()
        elif choice == 3:
            # Loop for multiple predictions
            clear_console()
            continue_running = True
            while continue_running:
                user_input_prediction(w, b)
                continue_running = wait_for_exit()
        
        print("\nYou are in the main menu")
        continue_running = wait_for_exit()
    

if __name__ == "__main__" :
    main()


