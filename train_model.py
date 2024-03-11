def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0 # - this is the overall loss 
        train_correct = 0
        train_total = 0
        
        total_iou = 0
        train_acc = 0

        # Training
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            #print("Inputs = ", inputs.shape, ", Label = ", labels.shape)

            optimizer.zero_grad()  # Обнуляем градиенты
            outputs = model(inputs)  # Получаем предсказания модели
            #print("Outputs = ", outputs.shape)
            loss = criterion(outputs, labels)  # Вычисляем функцию потерь
            loss.backward()  # Распространяем градиенты
            optimizer.step()  # Обновляем веса

            # - calculates the loss for the given batch (of size inputs.size(0)) and adds it to the overall loss
            train_loss += loss.item() * inputs.size(0)  
            # ????  the predicted tensor contains the predicted class indices for each pixel in the batch. ????
            o, predicted = torch.max(outputs , 1)
            # train_total accumulates the total number of samples processed so far in the current epoch
            train_total += labels.size(0)
            # computes the number of correct predictions in the current batch. 
            train_correct += (predicted==labels).sum().item()
            
            # calculate IoU for the current batch
            batch_iou = 0.0
            for pred_mask, true_mask in zip(predicted, labels):
                iou = calculate_iou(pred_mask.cpu(), true_mask.cpu())
                batch_iou += iou
                
            # calculate the averaged IoU for the current batch and accumulates total_iou over all batches
            total_iou += batch_iou / len(outputs)

        # normalizes the loss
        #train_loss = train_loss / len(train_loader.dataset)
        #train_loss_history.append(train_loss)
        '''
        When batches intersect, i.e., when some samples are present in multiple batches, simply dividing the accumulated loss by the total number of samples in the dataset (len(train_loader.dataset)) can lead to incorrect normalization. This is because the same samples may contribute to the loss multiple times, leading to an overestimation of the loss normalization factor.
        '''
        # normalizes the loss correctly
        # train_loss - loss for the given epoch
        train_loss = train_loss / train_total
        train_loss_history.append(train_loss)
        # normalizes the accuracy correctly - divide by the number of batches
        # train_acc - accuracy for the given epoch
        train_acc = total_iou / len(test_loader)
        train_acc_history.append(train_acc)

        

        # Validation
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs_val = model(inputs)
                loss = criterion(outputs_val, labels)
                val_loss += loss.item() * inputs.size(0)
                o, predicted_val = torch.max(outputs_val , 1)
                val_total += labels.size(0)
                val_correct += (predicted_val ==labels).sum().item()
                
                batch_iou = 0.0
                for pred_mask, true_mask in zip(predicted_val, labels):
                    iou = calculate_iou(pred_mask.cpu(), true_mask.cpu())
                    batch_iou += iou
                val_total_iou += batch_iou / len(outputs_val)

            #val_loss = val_loss / len(val_loader.dataset)
            # same problem as for training
            # correct normalisation:
            val_loss = val_loss / val_total
            val_loss_history.append(val_loss)
            
            val_acc = val_total_iou / len(val_loader)
            val_acc_history.append(val_acc)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    torch.save(model.state_dict(), const.model_out)

    history = {'train_loss': train_loss_history, 'val_loss': val_loss_history}
    plot_accuracy_and_loss(history)

    print('Training complete')
