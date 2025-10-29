melhorRedesPaths = {'melhor_rede_1.mat', 'melhor_rede_2.mat', 'melhor_rede_3.mat'};
classes = ["circle", "kite", "parallelogram", "square", "trapezoid", "triangle"];
imgSize = [28, 28];
numRepeticoes = 10;


[dataTeste, labelsTestes, targetsTeste] = carregarImagensFcn({'test'}, classes, imgSize);

resultadosCi = [];

for i = 1:length(melhorRedesPaths)
    load(melhorRedesPaths{i}, 'net');

    accs = zeros(1, numRepeticoes);
    confTotal = zeros(length(classes));

    for r = 1:numRepeticoes
        out = net(dataTeste);
        [~, pred] = max(out);
        [~, real] = max(targetsTeste);

        acc = sum(pred == real) / length(real) * 100;
        accs(r) = acc;
        confTotal = confTotal + confusionmat(real, pred);
    end

    accMedia = mean(accs);
    confMatMedia = round(confTotal / numRepeticoes);

    resultadosCi = [resultadosCi; {i, accMedia, evalc('disp(confMatMedia)')}];
end


T_ci = cell2table(resultadosCi, 'VariableNames', {'RedeID', 'PrecisaoTeste', 'MatrizConfusao'});
writetable(T_ci, 'resultados_c_i.xlsx');
