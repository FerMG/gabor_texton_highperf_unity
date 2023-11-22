function t = visualize_texton(filename, M, N)
%	t = visualize_texton(filename, M, N)
%   The output image t is a can be displayed with
%       imshow(uint8(t))

[order, mu, texton] = read_texton_file(filename);
Mt = size(texton,1);
Nt = size(texton,2);
nc = 3;

disp(Mt);
disp(Nt);

if(Mt>M||Nt>N)
    error('texton is larger than image visualization support');
end

% image constant to mean
disp(mu)
%t = repmat(mu, [M,N,1]);
t = repmat(mu*0, [M,N,1]);

for c=1:nc
    %u(1:Mt,1:Nt,c) = u(1:Mt,1:Nt,c) + sigma(c)/sigmatexton(c) * texton(:,:,c);
    t(1:Mt,1:Nt,c) = t(1:Mt,1:Nt,c) + sqrt((Mt-2*order)*(Nt-2*order)) * texton(:,:,c) + 0.5;
    disp(["fade channel ",c,": ", num2str(sqrt((Mt-2*order)*(Nt-2*order)))])
end
% translate texton to put in the center
%translate_vector = floor([M/2-Mt/2, N/2 - Nt/2, 0]);
%t = circshift(t, translate_vector);
imwrite (t, [filename,'.png'], "png");
%disp(t)
totalSumRGB = [sum(((t(:,:,1)-0.5)/256)(:)), sum(((t(:,:,2)-0.5)/256)(:)), sum(((t(:,:,3)-0.5)/256)(:))];
disp(totalSumRGB)
imshow(t);
%imshow(uint8(t));

end
