function [data, labels, targets] = carregarImagensFcn(pastas, classes, imgSize)

disp(['Carregando todas as imagens... {' strjoin(pastas, ', ') '}']);
data = [];
labels = strings(1, 0);

if ~iscell(pastas)
    error('pastas deve ser um cell array');
end
if ~isstring(classes)
    error('classes deve ser um array de strings');
end

for p = 1:length(pastas)
    currentFolder = pastas{p};

    for i = 1:length(classes)
        classDir = fullfile(currentFolder, classes(i));
        arquivos = dir(fullfile(classDir, '*.png'));

        for k = 1:length(arquivos)
            imgPath = fullfile(classDir, arquivos(k).name);
            img = imread(imgPath);
            img = im2gray(img);
            img = imbinarize(img);
            img = imresize(img, imgSize);

            data(:, end+1) = img(:);
            labels(end+1) = classes(i);
        end
    end
end

disp(['Total de imagens carregadas: ', num2str(size(data, 2))]);

% Create one-hot encoded targets
if nargout > 2
    targets = dummyvar(categorical(labels(:)))';
end
end