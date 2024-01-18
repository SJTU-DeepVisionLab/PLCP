function Q = Update_Q_com(P, Modeling, gama, train_p_com)


[p,q]=size(P);

options = optimoptions('quadprog','Display', 'off','Algorithm','interior-point-convex' );

Q = zeros(p,q);


for i = 1:p
	lb = train_p_com(i,:);
	ub = ones(q,1);
	Aeq = ub';
	beq = q-1;
	w = quadprog(2*eye(q), gama*P(i,:)-2*Modeling(i,:), [],[], Aeq, beq, lb, ub,[], options);
	Q(i,:) = w';
end

end