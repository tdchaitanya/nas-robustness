import torchattacks
from tqdm import tqdm
from net_utils import accuracy, AverageMeter, ProgressMeter
import time

def fgsm_attack_imnet(model, val_loader, epsilon):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    model.eval()
    # fgsm_attack = torchattacks.FGSM(model)
    # fgsm_attack = torchattacks.FGSM(model, eps=0.007)
    fgsm_attack = torchattacks.FGSM(model, eps=epsilon)
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
    # for i, (images, target) in enumerate(tqdm(val_loader)):
        images = images.cuda()
        images = fgsm_attack(images, target).cuda()
        target = target.cuda()
        output = model(images)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
    print(f'Test Accuracy on adversarial samples (FGSM): Top 1: {top1.avg}, Top 5: {top5.avg}')
    return top1.avg, top5.avg
    # print('Test Accuracy on adversarial samples: FGSM')
    # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
    #           .format(top1=top1, top5=top5))

def rfgsm_attack_imnet(model, val_loader, epsilon, alpha, iterations):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    model.eval()
    # rfgsm_attack = torchattacks.RFGSM(model)
    # rfgsm_attack = torchattacks.RFGSM(model, eps=0.007, alpha=0.03, iters=3)
    rfgsm_attack = torchattacks.RFGSM(model, eps=epsilon, alpha=alpha, iters=iterations)
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
    # for i, (images, target) in enumerate(tqdm(val_loader)):
        images = images.cuda()
        images = rfgsm_attack(images, target).cuda()
        target = target.cuda()
        output = model(images)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
    print(f'Test Accuracy on adversarial samples (R-FGSM): Top 1: {top1.avg}, Top 5: {top5.avg}')
    return top1.avg, top5.avg
    # print('Test Accuracy on adversarial samples: RFGSM')
    # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
    #           .format(top1=top1, top5=top5))

def stepll_attack_imnet(model, val_loader, epsilon, alpha, iterations):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    model.eval()
    # stepll_attack = torchattacks.StepLL(model)
    # stepll_attack = torchattacks.StepLL(model, eps=0.007, alpha=0.03, iters=3)
    stepll_attack = torchattacks.StepLL(model, eps=epsilon, alpha=alpha, iters=iterations)
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
    # for i, (images, target) in enumerate(tqdm(val_loader)):
        images = images.cuda()
        images = stepll_attack(images, target).cuda()
        target = target.cuda()
        output = model(images)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
    # print('Test Accuracy on adversarial samples: (Stepll)')
    # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
    #           .format(top1=top1, top5=top5))
    print(f'Test Accuracy on adversarial samples (StepLL): Top 1: {top1.avg}, Top 5: {top5.avg}')
    return top1.avg, top5.avg

def pgd_attack_imnet(model, val_loader, epsilon, alpha, iterations):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    model.eval()
    # pgd_attack = torchattacks.PGD(model, eps=0.05, alpha=0.01, iters=10)
    # pgd_attack = torchattacks.PGD(model, eps=0.007, alpha=0.03, iters=3)
    pgd_attack = torchattacks.PGD(model, eps=epsilon, alpha=alpha, iters=iterations)
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
    # for i, (images, target) in enumerate(tqdm(val_loader)):
        images = images.cuda()
        images = pgd_attack(images, target).cuda()
        target = target.cuda()
        output = model(images)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
    # print('Test Accuracy on adversarial samples: (PGD)')
    # print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
    #           .format(top1=top1, top5=top5))
    print(f'Test Accuracy on adversarial samples (PGD): Top 1: {top1.avg}, Top 5: {top5.avg}')
    return top1.avg, top5.avg
