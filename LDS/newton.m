function [x,fx,dfdx,xx] = newton(f,df,x0,TolX,MaxIter)

TolFun=eps;
xx(:,1) = x0; 
fx = feval(f,x0);

warning('off');
for k = 1:MaxIter
    dfdx = feval(df,xx(:,k));
    dx = -inv(dfdx)*fx;
    xx(:,k+1) = xx(:,k)+dx;
    fx = feval(f,xx(:,k+1));
    
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