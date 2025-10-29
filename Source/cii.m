melhorRedesPaths = {'melhor_rede_1.mat', 'melhor_rede_2.mat', 'melhor_rede_3.mat'};

numRepeticoes = 10;

% Configuração
pastas = {'test'};
pastasTestes = {'start', 'train', 'test'};
classes = ["circle", "kite", "parallelogram", "square", "trapezoid", "triangle"];
imgSize = [28, 28];
% ---------- ETAPA 1: Unir todas as imagens ----------
[dataALL, labelsALL, targetsALL] = carregarImagensFcn(pastas, classes, imgSize);

% ---------- ETAPA 2: Treinar redes com todos os dados ----------
disp('Iniciando treinamento das redes com todas as imagens...');
resultados = [];

for i = 1:length(melhorRedesPaths)
    disp(['Rede ', num2str(i), ': carregando...']);
    load(melhorRedesPaths{i}, 'net');

    % Treino completo
    net.divideFcn = 'dividetrain'; % usa todos os dados para treino
    disp(['Treinando rede ', num2str(i), ' com todos os dados...']);
    net = train(net, dataALL, targetsALL);

    % Salvar rede treinada
    nomeRedeTreinada = ['melhor_rede_totalii_' num2str(i) '.mat'];
    save(nomeRedeTreinada, 'net');
    disp(['Rede ', num2str(i), ' salva como ', nomeRedeTreinada]);

    % ---------- Precisão global ----------
    out = net(dataALL);
    [~, predGlobal] = max(out);
    [~, realGlobal] = max(targetsALL);
    accGlobal = sum(predGlobal == realGlobal) / length(realGlobal) * 100;
    disp(['Precisão global da rede ', num2str(i), ': ', num2str(accGlobal, '%.2f'), '%']);

    % ---------- Testes por pasta ----------
    for pt = 1:length(pastasTestes)
        pastaTeste = cellstr(pastasTestes{pt});
        disp(['Testando rede ', num2str(i), ' com imagens da pasta "', pastaTeste, '"...']);
        [dataALLTestes, labelsALLTestes, targetsALLTestes] = carregarImagensFcn(pastaTeste, classes, imgSize);

        accs = zeros(1, numRepeticoes);
        classAccsTotal = zeros(length(classes), numRepeticoes);
        confTotal = zeros(length(classes));

        for r = 1:numRepeticoes
            outTeste = net(dataALLTestes);
            [~, predTeste] = max(outTeste);
            [~, realTeste] = max(targetsALLTestes);

            acc = sum(predTeste == realTeste) / length(realTeste) * 100;
            confMat = confusionmat(realTeste, predTeste);
            classAcc = diag(confMat) ./ sum(confMat, 2) * 100;
            
            disp(['Repetição:', r, 'Precisão para "', pastaTeste, '": ', num2str(acc, '%.2f'), '%']);

            accs(r) = acc;
            classAccsTotal(:, r) = classAcc;
            confTotal = confTotal + confMat;
        end

        % Média dos resultados
        accTeste = mean(accs);
        classAcc = mean(classAccsTotal, 2);
        confMatMedia = round(confTotal / numRepeticoes);

        disp(['Repetição:', r, 'Precisão média para "', pastaTeste, '": ', num2str(accTeste, '%.2f'), '%']);

        [bestAcc, bestIdx] = max(classAcc);
        [worstAcc, worstIdx] = min(classAcc);

        bestClass = classes(bestIdx);
        worstClass = classes(worstIdx);


        resultados = [resultados; {i, accGlobal, pastaTeste, accTeste, evalc('disp(confMatMedia)'), classAcc', bestClass, bestAcc, worstClass, worstAcc}];

    end
end

% ---------- Exportar para Excel ----------
disp('Salvando resultados no arquivo "resultados_c_ii.xlsx"...');

T_final = cell2table(resultados, 'VariableNames', {
    'RedeID',
    'PrecisaoGlobal',
    'PastaTeste',
    'PrecisaoTeste',
    'MatrizConfusao',
    'ClassAcc',
    'MelhorForma',
    'MelhorAcc',
    'PiorForma',
    'PiorAcc'
    });
writetable(T_final, 'resultados_c_ii.xlsx', 'WriteMode', 'overwrite');
disp('Tudo concluído com sucesso!');