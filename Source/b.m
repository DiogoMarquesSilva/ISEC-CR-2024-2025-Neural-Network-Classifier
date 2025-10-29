clear; clc;

checkpointFile = 'checkpoint.mat';
if isfile(checkpointFile)
    load(checkpointFile);
    fprintf("Checkpoint loaded. Step = %d | idxRede = %d | t_start_idx = %d\n", step, idxRede, t_start_idx);
else
    resultados = [];
    melhoresRedes = {};
    idxRede = 1;
    t_start_idx = 1;
end

ticTotal = tic;

pathTrain = 'train';
classes = ["circle", "kite", "parallelogram", "square", "trapezoid", "triangle"];
numClasses = length(classes);
imgSize = [28, 28];

if ~exist('data', 'var')
    data = [];
    labels = strings(1, 0);

    for i = 1:length(classes)
        classe = classes(i);
        pastaClasse = fullfile(pathTrain, classe);
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
end

topologias = {5, 10, 20, [10 10], [20 20]};
funcoesAtivacaoOculta = {'tansig', 'logsig'};
funcoesAtivacaoSaida = {'purelin', 'logsig', 'softmax'};
funcoesTreino = {'trainlm', 'trainbfg', 'traingd'};
ratios = [
    0.7 0.15 0.15;
    0.6 0.2  0.2;
    0.50 0.25 0.25;
];

% Loop principal (retomável)
if ~exist('step', 'var')
    step = 1;
end

while step ~= 6

    combinacoes = {};
	if step == 1
		for t = 1:length(topologias)
			combinacoes{end+1} = {topologias{t}, 'tansig', 'purelin', 'trainlm', ratios(1,:)};
		end
	elseif step == 2
		for fAtivOculta = 1:length(funcoesAtivacaoOculta)
			combinacoes{end+1} = {bestTopo, funcoesAtivacaoOculta{fAtivOculta}, 'purelin', 'trainlm', ratios(1,:)};
		end
	elseif step == 3
		for fAtivSaida = 1:length(funcoesAtivacaoSaida)
			combinacoes{end+1} = {bestTopo, bestativOculta, funcoesAtivacaoSaida{fAtivSaida}, 'trainlm', ratios(1,:)};
		end
	elseif step == 4
		for fTreino = 1:length(funcoesTreino)
			combinacoes{end+1} = {bestTopo, bestativOculta, bestativSaida, funcoesTreino{fTreino}, ratios(1,:)};
		end
	elseif step == 5

		for r = 1:size(ratios,1)
			combinacoes{end+1} = {bestTopo, bestativOculta, bestativSaida, besttreino, ratios(r,:)};
        end
    end
    for i = t_start_idx:length(combinacoes)
        topo = combinacoes{i}{1};
        ativOculta = combinacoes{i}{2};
        ativSaida = combinacoes{i}{3};
        treino = combinacoes{i}{4};
        ratio = combinacoes{i}{5};
    
        fprintf("Topologia: %s | Oculta: %s | Saída: %s | Treino: %s | Ratio: %s\n", ...
            mat2str(topo), ativOculta, ativSaida, treino, mat2str(ratio));
    
        net = feedforwardnet(topo, treino);
        for l = 1:length(net.layers)
            if l < length(net.layers)
                net.layers{l}.transferFcn = ativOculta;
            else
                net.layers{l}.transferFcn = ativSaida;
            end
        end

    
        net.divideFcn = 'dividerand';
        net.divideParam.trainRatio = ratio(1);
        net.divideParam.valRatio = ratio(2);
        net.divideParam.testRatio = ratio(3);
        net.trainParam.showWindow = false;
        net.trainParam.epochs = 100;
    
        numRepeticoes = 10;
        accGlobalTemp = zeros(1, numRepeticoes);
        accTesteTemp = zeros(1, numRepeticoes);
        redesTemp = cell(1, numRepeticoes);
        confMatsTemp = cell(1, numRepeticoes);
    
        for rep = 1:numRepeticoes
            ticTreino = tic;
            [net, tr] = train(net, data, targets);
            
            out = net(data);
            [~, pred] = max(out);
            [~, real] = max(targets);
            accGlobal = sum(pred == real) / length(real) * 100;
    
            testInd = tr.testInd;
            [~, predTest] = max(out(:,testInd));
            [~, realTest] = max(targets(:,testInd));
            accTeste = sum(predTest == realTest) / length(realTest) * 100;
    
            accGlobalTemp(rep) = accGlobal;
            accTesteTemp(rep) = accTeste;
            redesTemp{rep} = net;
            confMatsTemp{rep} = confusionmat(realTest, predTest);
    
            fprintf("Repetição %d | Precisão global = %.2f %% | Precisão teste = %.2f %% | Tempo = %.2f s\n", ...
                rep, accGlobal, accTeste, toc(ticTreino));
        end
    
        mediaAccGlobal = mean(accGlobalTemp);
        mediaAccTeste = mean(accTesteTemp);
        [~, idxMelhorRede] = max(accTesteTemp);
        confStr = evalc('disp(confMatsTemp{idxMelhorRede})');

        resultados = [resultados; {
            mat2str(topo), ativOculta, ativSaida, treino, mat2str(ratio), ...
            mediaAccGlobal, mediaAccTeste, confStr, redesTemp{idxMelhorRede}
        }];

    
        % Verificar se a configuração já existe
        configAtual = struct( ...
            "Topologia", topo, ...
            "FuncAtivOculta", ativOculta, ...
            "FuncAtivSaida", ativSaida, ...
            "FuncTreino", treino, ...
            "Ratio", ratio ...
        );
        
        isDuplicate = false;
        for k = 1:length(melhoresRedes)
            cfg = melhoresRedes{k}.config;
            if isequal(cfg.Topologia, configAtual.Topologia) && ...
               strcmp(cfg.FuncAtivOculta, configAtual.FuncAtivOculta) && ...
               strcmp(cfg.FuncAtivSaida, configAtual.FuncAtivSaida) && ...
               strcmp(cfg.FuncTreino, configAtual.FuncTreino) && ...
               isequal(cfg.Ratio, configAtual.Ratio)
                isDuplicate = true;
                % Substitui se a nova rede for melhor
                if mediaAccTeste > melhoresRedes{k}.acc
                    melhoresRedes{k}.net = redesTemp{idxMelhorRede};
                    melhoresRedes{k}.acc = mediaAccTeste;
                end
                break;
            end
        end
        
        % Se não for duplicada, verifica se cabe entre as 3 melhores
        if ~isDuplicate
            if length(melhoresRedes) < 3
                melhoresRedes{end+1} = struct("net", redesTemp{idxMelhorRede}, "acc", mediaAccTeste, "config", configAtual);
            else
                accValues = [melhoresRedes{:}];
                accValues = [accValues.acc];
                [~, idxMin] = min(accValues);
                if mediaAccTeste > melhoresRedes{idxMin}.acc
                    melhoresRedes{idxMin} = struct("net", redesTemp{idxMelhorRede}, "acc", mediaAccTeste, "config", configAtual);
                end
            end
        end


    
        idxRede = idxRede + 1;
        t_start_idx = i + 1;
    
        
        save(checkpointFile);
        fprintf("Progresso salvo após a combinação %d\n\n", i);
    end
    step = step + 1;
    t_start_idx = 1;
    [~, idxMelhor] = max(cell2mat(resultados(end-length(combinacoes)+1:end, 7)));
    melhorComb = combinacoes{idxMelhor};
    
    % Atualizar os parâmetros para o próximo passo
    bestTopo = melhorComb{1};
    bestativOculta = melhorComb{2};
    bestativSaida = melhorComb{3};
    besttreino = melhorComb{4};
    bestratio = melhorComb{5};
    
    fprintf("\nFim da etapa %d:\n", step - 1);
    fprintf("Melhor combinação → Topologia: %s | Oculta: %s | Saída: %s | Treino: %s | Ratio: %s\n", ...
        mat2str(bestTopo), bestativOculta, bestativSaida, besttreino, mat2str(bestratio));
    fprintf("Precisão média na validação/teste: %.2f %%\n", cell2mat(resultados(end-length(combinacoes)+idxMelhor, 7)));
    fprintf("============================================================\n\n");


end


% ---------- Exportar para Excel ----------
T = cell2table(resultados, ...
    'VariableNames', {'Topologia', 'FuncAtivOculta', 'FuncAtivSaida', ...
                      'FuncTreino', 'Ratio', 'PrecisaoGlobal', 'PrecisaoTeste', ...
                      'MatrizConfusao', 'Rede'});

writetable(removevars(T, 'Rede'), 'resultados_b.xlsx');
disp(T(:,1:7))

% ---------- Guardar as melhores redes----------
for i = 1:length(melhoresRedes)
    nome = sprintf('melhor_rede_%d.mat', i);
    net = melhoresRedes{i}.net;
    config = melhoresRedes{i}.config;
    acc = melhoresRedes{i}.acc;
    save(nome, 'net', 'config', 'acc');
    
    fprintf("✔️  Rede salva: %s\n", nome);
    fprintf("   ↳ Topologia: %s | Oculta: %s | Saída: %s | Treino: %s | Ratio: %s | Acurácia: %.2f%%\n", ...
        mat2str(config.Topologia), config.FuncAtivOculta, config.FuncAtivSaida, ...
        config.FuncTreino, mat2str(config.Ratio), acc);
end



fprintf("\n→ Tempo total da execução: %.2f segundos\n", toc(ticTotal));
