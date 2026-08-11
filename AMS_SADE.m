classdef AMS_SADE < ALGORITHM
% <2026> <single> <real> <large/none> <expensive>
% AMS-SADE: Adaptive Multi-Swarm Surrogate-Assisted Differential Evolution

    methods
        function main(Algorithm, Problem)

            %% Basic Configuration & Initialization
            warning('off', 'all');
            D     = Problem.D;
            maxFE = Problem.maxFE;
            lower = Problem.lower;
            upper = Problem.upper;
            range = upper - lower + 1e-12;

            N_init = (D < 100) * 100 + (D >= 100) * 150;
            N_init = min(N_init, maxFE - 1);
            N_init = max(N_init, min(20, maxFE - 1));

            PopDec = lower + range .* lhsdesign(N_init, D);
            Archive = Problem.Evaluation(PopDec);

            iter      = 0;
            maxIter   = max(1, maxFE - N_init);
            noImprove = 0;
            sigma     = 0.20; 
            
            nCand = (D <= 100)*1000 + (D > 100 && D <= 500)*3000 + (D > 500)*5000;
            sub_weights = [0.4, 0.3, 0.3]; 
            
            H = 6; MemF = 0.5*ones(H,1); MemCR = 0.5*ones(H,1); k_mem = 1;

            %% Main Closed-Loop Optimization
            while Algorithm.NotTerminated(Archive)
                iter = iter + 1;
                FE_ratio = iter / maxIter;

                X = double(Archive.decs);
                Y = double(Archive.objs);
                N = size(X,1);

                [Ysort, rank] = sort(Y);
                BestX = X(rank(1), :);
                BestF = Ysort(1);

                Xn = (X - lower) ./ range;
                BestXn = (BestX - lower) ./ range;

                %% [Innovation 1] Fast Local RBF Agents
                % Abandon global modeling to eliminate high-dimensional phantom overhead
                K_loc = min(150, N);
                loc_idx = rank(1:K_loc);
                local_model = build_rbf_fast(Xn(loc_idx, :), Y(loc_idx));

                N1 = max(10, round(sub_weights(1) * nCand));
                N2 = max(10, round(sub_weights(2) * nCand));
                N3 = nCand - N1 - N2; 

                %% [Innovation 2] Heterogeneous Multi-Swarm Co-evolution
                % Coordinate three complementary swarms to balance exploration and exploitation
                
                % --- Swarm 1: Sparse Exploration ---
                p0 = min(20 / D, 1);
                p  = p0 * (1 - log(max(1,iter)) / log(max(2,maxIter)));
                p  = max(p, 1 / D);

                Cand1 = repmat(BestXn, N1, 1);
                mask = rand(N1, D) < p;
                emptyRow = ~any(mask,2);
                if any(emptyRow)
                    id = find(emptyRow);
                    for ii = 1:length(id); mask(id(ii), randi(D)) = true; end
                end
                noise = sigma * randn(N1, D);
                Cand1(mask) = Cand1(mask) + noise(mask);

                % --- Swarm 2: Adaptive Manifold Contraction ---
                Cand2 = zeros(N2, D);
                eliteNum = min(max(10, round(0.15*N)), N);
                eliteIdx = rank(1:eliteNum);
                for i = 1:N2
                    mid = randi(H);
                    F = min(1.0, max(0.1, MemF(mid) + 0.1*tan(pi*(rand-0.5))));
                    CR = min(1.0, max(0.0, MemCR(mid) + 0.1*randn));
                    
                    xi = Xn(eliteIdx(randi(eliteNum)), :);
                    xp = Xn(eliteIdx(randi(eliteNum)), :);
                    r1 = randi(N); r2 = randi(N);
                    mutant = xi + F*(xp - xi) + F*(Xn(r1,:) - Xn(r2,:));
                    
                    cross = rand(1,D) < CR;
                    if ~any(cross); cross(randi(D)) = true; end
                    trial = xi; trial(cross) = mutant(cross);
                    
                    alpha = 0.60 + 0.35 * FE_ratio; 
                    Cand2(i,:) = alpha * BestXn + (1-alpha) * trial;
                end

                % --- Swarm 3: Agent Gradient Guidance ---
                delta = 1e-5;
                Pred_center = predict_rbf_fast(BestXn, local_model);
                
                FD_X = repmat(BestXn, D, 1) + delta * eye(D);
                Pred_FD = predict_rbf_fast(FD_X, local_model);
                Grad = (Pred_FD - Pred_center)' / delta;
                
                Grad_norm = norm(Grad) + 1e-12;
                Direction = Grad / Grad_norm;
                
                step_sizes = linspace(-0.2, 1.5, N3)' * sigma; 
                Cand3 = repmat(BestXn, N3, 1) - step_sizes * Direction;

                %% [Innovation 1-b] 1-FE Merit-Based Pre-screening
                Cand = [Cand1; Cand2; Cand3];
                Cand = min(1, max(0, Cand));
                Source = [ones(N1,1); 2*ones(N2,1); 3*ones(N3,1)];
             
                % Fast algebraic distance computation
                C2 = sum(Cand.^2, 2);
                X2 = sum(Xn.^2, 2)';
                D2 = max(0, bsxfun(@plus, C2, X2) - 2 * (Cand * Xn'));
                distToArchive = sqrt(min(D2, [], 2));
                
                valid = distToArchive > 1e-10;
                
                if sum(valid) < 10
                    Cand_final = Cand; dist_final = distToArchive; Source_final = Source;
                else
                    Cand_final = Cand(valid,:); dist_final = distToArchive(valid); Source_final = Source(valid);
                end
                Pred = predict_rbf_fast(Cand_final, local_model);
                
                pred_s = normalize_minmax(Pred);
                dist_s = normalize_minmax(dist_final);
                w = 0.85 + 0.14 * FE_ratio;
                merit = w * pred_s - (1-w) * dist_s;
                [~, bestCandIdx] = min(merit);
                
                NewDec = lower + Cand_final(bestCandIdx,:) .* range;
                NewDec = min(upper, max(lower, NewDec));
                winning_source = Source_final(bestCandIdx);

                %% True Evaluation (1 FE)
                NewSol = Problem.Evaluation(NewDec);
                Archive = [Archive, NewSol];
                NewF = double(NewSol.obj);

                %% [Innovation 3] Dynamic Resource Allocation (DRA)
                % Closed-loop feedback mechanism to adjust weights and exploration steps
                if NewF < BestF - 1e-12
                    sigma = min(0.30, sigma * 1.05);
                    noImprove = 0;
                    
                    sub_weights(winning_source) = sub_weights(winning_source) + 0.05;
                    
                    if winning_source == 2
                        MemF(k_mem) = 0.9*MemF(k_mem) + 0.1*0.5;
                        MemCR(k_mem) = 0.9*MemCR(k_mem) + 0.1*0.5;
                        k_mem = mod(k_mem, H) + 1;
                    end
                else
                    sigma = max(0.001, sigma * 0.94);
                    noImprove = noImprove + 1;
                    sub_weights(winning_source) = max(0.1, sub_weights(winning_source) - 0.01);
                end
                
                sub_weights = sub_weights / sum(sub_weights);
                
                % Anti-stagnation reset
                if noImprove >= 25
                    sigma = 0.20; 
                    noImprove = 0; 
                    sub_weights = [0.34, 0.33, 0.33]; 
                end
            end
        end
    end
end

%% --- Fast Cubic RBF Surrogate Model ---
function model = build_rbf_fast(X, Y)
    X = double(X); Y = double(Y(:));
    [X, ia] = unique(X, 'rows', 'stable'); Y = Y(ia);
    N = size(X,1);
    Ymin = min(Y); Yrng = max(1e-12, max(Y)-Ymin);
    Yn = (Y - Ymin) ./ Yrng;
    
    X2 = sum(X.^2, 2);
    D2 = max(0, bsxfun(@plus, X2, X2') - 2 * (X * X'));
    Phi = D2.^1.5; 
    P = ones(N,1);
    
    solved = false;
    for lam = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1]
        A = [Phi + lam*eye(N), P; P', 0]; b = [Yn; 0];
        if rcond(A) > 1e-14
            warning('off','all'); coef = A \ b; warning('on','all');
            solved = true; break;
        end
    end
    if ~solved; A = [Phi + 0.01*eye(N), P; P', 0]; coef = pinv(A)*[Yn; 0]; end
    model.X = X; model.lambda = coef(1:N); model.gamma = coef(N+1);
    model.Ymin = Ymin; model.Yrng = Yrng;
end

function Yp = predict_rbf_fast(X, model)
    X = double(X);
    X2 = sum(X.^2, 2);
    M2 = sum(model.X.^2, 2)';
    D2 = max(0, bsxfun(@plus, X2, M2) - 2 * (X * model.X'));
    Phi = D2.^1.5; 
    
    Yn = Phi * model.lambda + model.gamma;
    Yp = Yn * model.Yrng + model.Ymin;
    Yp = Yp(:);
end

function z = normalize_minmax(x)
    x = double(x(:)); xmin = min(x); xmax = max(x);
    if xmax - xmin < 1e-12; z = zeros(size(x));
    else; z = (x - xmin) ./ (xmax - xmin); end
end