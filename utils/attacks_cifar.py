import torchattacks
from tqdm import tqdm


def fgsm_attack_cifar(model, testloader, epsilon):
    model.eval()
    correct = 0
    total = 0
    # fgsm_attack = torchattacks.FGSM(model, eps=0.007)
    fgsm_attack = torchattacks.FGSM(model, eps=epsilon)
    for batch_idx, (images, labels) in enumerate(testloader):
    # for batch_idx, (images, labels) in enumerate(tqdm(testloader)):
        images = fgsm_attack(images, labels).cuda()
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.cuda()).sum().item()
    acc = 100 * float(correct)/total
    print(f'Test Accuracy on adversarial samples (FGSM): {acc}')
    return acc

def pgd_attack_cifar(model, testloader, epsilon, alpha, iterations):#, n_restarts):
    model.eval()
    correct = 0
    total = 0
    pgd_attack = torchattacks.PGD(model, eps=epsilon, alpha=alpha, steps=iterations, random_start=True)

    for batch_idx, (images, labels) in enumerate(testloader):
        images = pgd_attack(images, labels).cuda()
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.cuda()).sum().item()
    
    acc = 100 * float(correct)/total
    # print(f'Test Accuracy on adversarial samples (PGD): {acc}')
    return acc

def rfgsm_attack_cifar(model, testloader, epsilon, alpha, iterations):
    model.eval()
    correct = 0
    total = 0
    # rfgsm_attack = torchattacks.RFGSM(model)
    # rfgsm_attack = torchattacks.RFGSM(model, eps=0.007, alpha=0.03, iters=3)
    rfgsm_attack = torchattacks.RFGSM(model, eps=epsilon, alpha=alpha, steps=iterations)
    for batch_idx, (images, labels) in enumerate(testloader):
    # for batch_idx, (images, labels) in enumerate(tqdm(testloader)):
        images = rfgsm_attack(images, labels).cuda()
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.cuda()).sum().item()
    acc = 100 * float(correct)/total
    print(f'Test Accuracy on adversarial samples (R-FGSM): {acc}')
    return acc

# def stepll_attack_cifar(model, testloader, epsilon, alpha, iterations):
#     model.eval()
#     correct = 0
#     total = 0
#     # stepll_attack = torchattacks.StepLL(model)
#     # stepll_attack = torchattacks.StepLL(model, eps=0.007, alpha=0.03, iters=3)
#     stepll_attack = torchattacks.StepLL(model, eps=epsilon, alpha=alpha, iters=iterations)
#     for batch_idx, (images, labels) in enumerate(testloader):
#     # for batch_idx, (images, labels) in enumerate(tqdm(testloader)):
#         images = stepll_attack(images, labels).cuda()
#         outputs = model(images)
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels.cuda()).sum().item()
#     acc = 100 * float(correct)/total
#     print(f'Test Accuracy on adversarial samples (Step-LL): {acc}')
#     return acc

def fab_attack_cifar(model, testloader, epsilon, iterations, n_classes=10):   #, epsilon
    model.eval()
    correct = 0
    total = 0
    
    fab_attack = torchattacks.FAB(model, eps=epsilon, steps=iterations, n_classes=n_classes)    #, eps=epsilon
    
    for batch_idx, (images, labels) in enumerate(testloader):
        images = fab_attack(images, labels).cuda()
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.cuda()).sum().item()
    acc = 100 * float(correct)/total
    print(f'Test Accuracy on adversarial samples (FAB): {acc}')
    return acc

def square_attack_cifar(model, testloader, epsilon):   #, epsilon
    model.eval()
    correct = 0
    total = 0
    
    square_attack = torchattacks.Square(model, eps=epsilon)    #, eps=epsilon
    
    for batch_idx, (images, labels) in enumerate(testloader):
        images = square_attack(images, labels).cuda()
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.cuda()).sum().item()
    acc = 100 * float(correct)/total
    print(f'Test Accuracy on adversarial samples (Square): {acc}')
    return acc

def apgd_attack_cifar(model, testloader, epsilon, iterations, n_restarts):   #, epsilon
    model.eval()
    correct = 0
    total = 0
    
    apgd_attack = torchattacks.APGD(model, eps=epsilon, steps=iterations, n_restarts=n_restarts)    #, eps=epsilon
    
    for batch_idx, (images, labels) in enumerate(testloader):
        images = apgd_attack(images, labels).cuda()
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.cuda()).sum().item()
    acc = 100 * float(correct)/total
    # print(f'Test Accuracy on adversarial samples (APGD): {acc}')
    return acc

def apgdt_attack_cifar(model, testloader, epsilon, n_classes=10):   #, epsilon
    model.eval()
    correct = 0
    total = 0
    
    apgdt_attack = torchattacks.APGDT(model, eps=epsilon, n_classes=n_classes)    #, eps=epsilon
    
    for batch_idx, (images, labels) in enumerate(testloader):
        images = apgdt_attack(images, labels).cuda()
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.cuda()).sum().item()
    acc = 100 * float(correct)/total
    print(f'Test Accuracy on adversarial samples (APGDT): {acc}')
    return acc

def auto_attack_cifar(model, testloader, epsilon, n_classes=10):   #, epsilon
    model.eval()
    correct = 0
    total = 0
    
    auto_attack = torchattacks.AutoAttack(model, eps=epsilon, n_classes=n_classes)    #, eps=epsilon
    
    for batch_idx, (images, labels) in enumerate(testloader):
        images = auto_attack(images, labels).cuda()
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.cuda()).sum().item()
    acc = 100 * float(correct)/total
    print(f'Test Accuracy on adversarial samples (AutoAttack): {acc}')
    return acc
