function v = nmi_jwy(label_gt, label_pre)
% Standard version of Nomalized mutual information which can cope with different number of clusters between label_gt and label_pre
% Written by Jianwei Yang (jw2yang@vt.edu). August 2015.
size = length(label_gt);

%% entropy_gt
label_gt_unique = unique(label_gt);
hist_label_gt = zeros(1, length(label_gt_unique));
for i = 1:length(label_gt_unique)
    hist_label_gt(i) = sum(label_gt == label_gt_unique(i));
end

% calculate entropy for label_gt
ent_gt = 0;
for i = 1:length(label_gt_unique)
    ent_gt = ent_gt + hist_label_gt(i) * log(hist_label_gt(i) / size);
end

%% entropy_pred
idx = label_pre;
label_pred_unique = unique(idx);
hist_label_pred = zeros(1, length(label_pred_unique));
for i = 1:length(label_pred_unique)
    hist_label_pred(i) = sum(idx == label_pred_unique(i));
end

% calculate entropy for label_pred
ent_pred = 0;
for i = 1:length(hist_label_pred)
    ent_pred = ent_pred + hist_label_pred(i) * log(hist_label_pred(i) / size);
end

%% mutual information
mutual_info = 0;
for i = 1:length(label_gt_unique)
    labels_i = find(label_gt == label_gt_unique(i));
    for j = 1:length(label_pred_unique)
        labels_j = find(idx == label_pred_unique(j));
        n_h_l = 0;
        for m = 1:length(labels_i)
            for n = 1:length(labels_j)
                if labels_i(m) == labels_j(n)
                    n_h_l = n_h_l + 1;
                end
            end
        end
        
        if n_h_l == 0
            continue;
        end        
        mutual_info = mutual_info + ...
        n_h_l * log(size * n_h_l / length(labels_i) / length(labels_j));
    end
end
% mutual_info
v = mutual_info / sqrt(ent_gt * ent_pred);
