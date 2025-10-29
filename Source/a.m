clear; clc;

pathStart = 'start';
classes = ["circle", "kite", "parallelogram", "square", "trapezoid", "triangle"];
numClasses = length(classes);
imgSize = [28, 28];

data = [];
labels = strings(1, 0);

for i = 1:length(classes)
    classe = classes(i);
    pastaClasse = fullfile(pathStart, classe);
    imagens = dir(fullfile(pastaClasse, '*.png'));  
    for j = 1:length(imagens)
        img = imread(fullfile(pastaClasse, imagens(j).name));

        img = im2gray(img);
        img = imbinarize(img);
        img = imresize(img, imgSize);
        data = [data; img(:)'];
        labels(end+1) = classe;
    end
end

fprintf("i) Total de imagens carregadas: %d\n", size(data, 1));

data = data';                         
targets = dummyvar(categorical(labels'))';  

% DEFINIR TOPOLOGIAS A TESTAR
topologias = {
    5,
    10,        
    20,     
    [10 10]  
    [20 20]
};

numReps = 10;
resultados = [];
precisoesDetalhadas = [];  

tic
for t = 1:length(topologias)
    topo = topologias{t};
    precisoes = zeros(1, numReps);

    for r = 1:numReps
        fprintf("Topologia: %s | Repetição: %d\n", mat2str(topo), r);
        net = feedforwardnet(topo);
        net.trainParam.showWindow = false;
        net.trainParam.epochs = 50;
        net.divideParam.trainRatio = 1;
        net.divideParam.valRatio = 0;
        net.divideParam.testRatio = 0;

        trainTime = tic;
        net = train(net, data, targets);
        toc(trainTime);

        out = net(data);
        [~, pred] = max(out);
        [~, real] = max(targets);
        acc = sum(pred == real) / length(real) * 100;
        precisoes(r) = acc;



       
        precisoesDetalhadas = [precisoesDetalhadas; {mat2str(topo), r, acc}];
    end

    media = mean(precisoes);
    desvio = std(precisoes);
    resultados = [resultados; {mat2str(topo), media, desvio}];
end
toc

T = cell2table(resultados, 'VariableNames', {'Topologia', 'MediaPrecisao', 'DesvioPadrao'});
disp(T);
writetable(T, 'resultados_a.xlsx');
