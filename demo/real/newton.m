function [x,fx,dfdx,xx] = newton(f,df,x0,TolX,MaxIter)

TolFun=eps;
xx(:,1) = x0; 
fx = feval(f,x0);
% disp(norm(fx))
warning('off');
for k = 1:MaxIter
%     disp(k)
    dfdx = feval(df,xx(:,k));
    dx = -dfdx\fx;
    xx(:,k+1) = xx(:,k)+dx;
    fx = feval(f,xx(:,k+1));
    
    if(sum(isnan(xx(:,k+1)))>0)
        break;
    end
%     disp(norm(fx))
    if(norm(fx)<TolFun || norm(dx) < TolX)
        break;
    end
end
warning('on');

x = xx(:,k + 1);
dfdx = feval(df,x);
if(k == MaxIter)
    fprintf('The best in %d iterations\n',MaxIter)
end

end