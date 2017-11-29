python_return_value = py.bbd.main('ABSOLUTE_PATH_OF_THE_IMAGE\images\organic.tif');
python_arrays = python_return_value{1};

facets_number = size(python_arrays, 2);
matlab_cell = cell(facets_number,1);

for i = 1:facets_number
    temp_array = python_arrays{i};
    matlab_cell{i} = double(py.array.array('d',py.numpy.nditer(temp_array)));
end