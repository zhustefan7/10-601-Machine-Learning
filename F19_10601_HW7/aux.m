train_accuracy = [0.8328168509141984, 0.8336861129254841, 0.8607658345651971, 0.9378649549899077];
test_accuracy = [0.8325026284023208, 0.8335539893306335, 0.8564697636384876, 0.9225692145944473];

x = [10,100,1000,10000]
figure();
plot(x,train_accuracy);hold on;
plot(x,test_accuracy);
legend('Training Accuracy' , 'Testing Accuracy')
xlabel('Sequences used for trainig')
ylabel('Accuracy')