% ---------- ETAPA D: Testar desenhos manuais ----------
disp('Testando desenhos manuais nas melhores redes...');

% Pasta dos desenhos manuais
pastaDesenhos = {'desenhos'};
classes = ["circle", "kite", "parallelogram", "square", "trapezoid", "triangle"];
imgSize = [28 28];
numRepeticoes = 10;

% Carregar dados dos desenhos
[dataDesenhos, labelsDesenhos, targetsDesenhos] = carregarImagensFcn(pastaDesenhos, classes, imgSize);

% Verificação de consistência
if isempty(dataDesenhos)
    error('Nenhuma imagem foi carregada da pasta "desenhos". Verifique os diretórios e formatos.');
end

% Testar nas redes
resultadosDesenhos = [];

for i = 1:length(melhorRedesPaths)
    disp(['Carregando rede ', num2str(i), '...']);
    load(['melhor_rede_totaliii_' num2str(i) '.mat'], 'net');

    accs = zeros(1, numRepeticoes);
    classAccsTotal = zeros(length(classes), numRepeticoes);
    confTotal = zeros(length(classes));

    for r = 1:numRepeticoes
        outDesenhos = net(dataDesenhos);
        [~, predDesenhos] = max(outDesenhos);
        [~, realDesenhos] = max(targetsDesenhos);

        acc = sum(predDesenhos == realDesenhos) / length(realDesenhos) * 100;
        confMat = confusionmat(realDesenhos, predDesenhos);
        classAcc = diag(confMat) ./ sum(confMat, 2) * 100;

        accs(r) = acc;
        classAccsTotal(:, r) = classAcc;
        confTotal = confTotal + confMat;
    end

    accDesenhos = mean(accs);
    classAccDesenhos = mean(classAccsTotal, 2);
    confMatMedia = round(confTotal / numRepeticoes);

    [bestAcc, bestIdx] = max(classAccDesenhos);
    [worstAcc, worstIdx] = min(classAccDesenhos);

    bestClass = classes{bestIdx};
    worstClass = classes{worstIdx};

    resultadosDesenhos = [resultadosDesenhos; {
        i, ...
        accDesenhos, ...
        evalc('disp(confMatMedia)'), ...
        classAccDesenhos', ...
        bestClass, ...
        bestAcc, ...
        worstClass, ...
        worstAcc
    }];

    disp(['Média de precisão dos desenhos na rede ', num2str(i), ': ', num2str(accDesenhos, '%.2f'), '%']);
end

% Exportar resultados
T_desenhos = cell2table(resultadosDesenhos, 'VariableNames', {
    'RedeID',
    'Precisao',
    'MatrizConfusaoMedia',
    'PrecisaoPorClasse',
    'MelhorClasse',
    'MelhorPrecisao',
    'PiorClasse',
    'PiorPrecisao'
});
writetable(T_desenhos, 'resultados_d.xlsx', 'WriteMode', 'overwrite');
disp('Resultados médios dos desenhos salvos em "resultados_d.xlsx".');
