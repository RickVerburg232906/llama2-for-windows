import yaml
import curses

def get_model_selection(stdscr):
    models = [
        "Llama-2-7b-hf",
        "Llama-2-7b-chat-hf",
        "Llama-2-13b-hf",
        "Llama-2-13b-chat-hf",
        "Llama-2-70b-hf",
        "Llama-2-70b-chat-hf"
    ]
    
    current_idx = 0
    while True:
        stdscr.addstr(0, 0, "Select a model using arrow keys and press Enter:\nModel Suggested: Llama-2-7b-chat-hf")
        for idx, model in enumerate(models):
            if idx == current_idx:
                stdscr.addstr(idx + 2, 0, "-> " + model, curses.A_BOLD)
            else:
                stdscr.addstr(idx + 2, 0, "   " + model)
        key = stdscr.getch()
        if key == curses.KEY_UP and current_idx > 0:
            current_idx -= 1
        elif key == curses.KEY_DOWN and current_idx < len(models) - 1:
            current_idx += 1
        elif key == 10:  # Enter key
            break
    return models[current_idx]


def main():
    # Collect information

    print("======================================")
    print("   Welcome to llama2-for-windows")
    print("======================================")
    print("\nPlease follow the instructions below:\n")

    batch_size = input("Enter batch size (21->1000): ")
    while not (batch_size.isdigit() and 21 <= int(batch_size) <= 1000):
        print("Invalid batch size. Please provide a value between 21 and 1000.")
        print("Suggestion: 25")
        batch_size = input("Enter batch size (21->1000): ")

    token = input("Enter your Hugging Face token: ")

    # Select model
    model_selected = curses.wrapper(get_model_selection)

    # Construct the config dict
    config = {
        'general': {
            'logging_level': 'WARNING',
            'pytorch_cuda_config': f'max_split_size_mb:{batch_size}'
        },
        'model': {
            'token': token,
            'id': f'meta-llama/{model_selected}'
        }
    }

    # Save to yaml
    with open('config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    print("config.yaml has been saved!")
    print("You can now execute < .\llama2.py > or < .\llama2-web.py > to start!")

if __name__ == "__main__":
    main()
