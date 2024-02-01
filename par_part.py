if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((90,220)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    lbl = 0
    num_classes = 11
    save_path = 'models/best_model_uc_alexnet_batch_mod_asym.pth'

    train_set = CustomDataset("training_set","training_set.txt", transform, target_label=lbl)
    val_set = CustomDataset("validation_set","validation_set.txt", transform, target_label=lbl, shuffle=False)

    lossFun = nn.CrossEntropyLoss()
    from asym_loss import ASLSingleLabel
    lossFun = ASLSingleLabel(gamma_pos=torch.zeros(1,11),gamma_neg=torch.tensor([1, 2, 3, 2, 3, 4, 4, 4, 3, 2, 4]))

    model = SimpleModel(num_classes, 'alexnet')
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    num_epochs = 30
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("DEVICE:", device)
    model.to(device)

    val_loss_min = float('inf')
    early_stopping_patience = 10
    early_stopping_counter = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_batch_size, val_batch_size = 134, 16
        val_loader = DataLoader(val_set, batch_size=val_batch_size)
        train_loader = DataLoader(train_set, batch_size=train_batch_size)
        model.train()
        running_loss = 0.0
        loss = 0
        correct = 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = lossFun(outputs, labels)
            correct += ((outputs.argmax(dim=1)) == labels).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_acc = correct

        val_loss = 0
        correct = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                correct += ((outputs.argmax(dim=1)) == labels).sum().item()
                loss = lossFun(outputs, labels)
                val_loss += loss.item()

        scheduler.step(val_loss)

        if val_loss < val_loss_min:
            val_loss_min = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                }, save_path)
            print('SAVED best model')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

        print("TRAIN LOSS EPOCH:", epoch, ":", running_loss*train_batch_size/len(train_set),"ACC:",train_acc/len(train_set))
        print("VAL LOSS EPOCH:", epoch, ":", val_loss*val_batch_size/len(val_set),"ACC:",correct/len(val_set))