import time
import torch
from tqdm import tqdm
from weight_train_utils import get_time_dif, eval_dataset


def train(config, model, llm_model, input_builder, train_loader, dev_loader):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for train_data in train_loader:
            w1, w2 = model()
            model.zero_grad()
            results = []
            for data in tqdm(train_data):
                input = input_builder.process_input(data, w1, w2)
                prediction = llm_model.generate_sentence(input)
                result = {
                    "prediction": prediction,
                    "ground_truth": data['answer'],
                }
                results.append(result)
            acc = eval_dataset(results)
            loss = (1 - acc) * 10 + w1 * 1e-5
            loss.backward()
            optimizer.step()

            if total_batch % config.eval_steps == 0:
                dev_loss, dev_acc = evaluate(model, llm_model, input_builder, dev_loader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    w1, w2 = model()
                    output = "w1: " + str(w1.item()) + " w2: " + str(w2.item()) + " train_acc: " + str(
                        acc * 100) + " dev_acc: " + str(dev_acc * 100) + "\n"
                    with open(config.output_path, 'w') as f:
                        f.write(output)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''

                w1, w2 = model()
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  w1: {1:>5.2},  w2: {2:>5.2},  Train Loss: {3:>5.2},  Train Acc: {4:>6.2%},  Val Loss: {5:>5.2},  Val Acc: {6:>6.2%},  Time: {7} {8}'
                print(
                    msg.format(total_batch, w1.item(), w2.item(), loss.item(), acc, dev_loss, dev_acc, time_dif,
                               improve))
                model.train()
            total_batch += 1

            if total_batch - last_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break

        if flag:
            break


def evaluate(model, llm_model, input_builder, dev_loader):
    model.eval()
    loss_total = 0
    acc_total = 0

    for eval_data in dev_loader:
        w1, w2 = model()
        results = []
        for data in tqdm(eval_data):
            input = input_builder.process_input(data, w1, w2)
            prediction = llm_model.generate_sentence(input)
            result = {
                "prediction": prediction,
                "ground_truth": data['answer'],
            }
            results.append(result)
        acc = eval_dataset(results)
        loss = (1 - acc) * 10 + w1 * 1e-5
        loss_total += loss.item()
        acc_total += acc

    return loss_total / len(dev_loader), acc_total / len(dev_loader)
