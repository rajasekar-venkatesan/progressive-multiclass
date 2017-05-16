function [TestingAccuracy] = Test_result(Test_input,input_weight,bias,beta,Test_output,label)

MissClassificationRate_Testing = 0;
test_output_bipolar = to_bipolar(Test_output,label);
test_H = SigActFun(Test_input,input_weight,bias);
test_output = test_H * beta;
nTestingData=size(Test_input,1);
    for i = 1 : nTestingData
        [x, label_index_expected]=max(test_output_bipolar(i,:));
        [x, label_index_actual]=max(test_output(i,:));
        expect_actual_test(i,:) = [Test_output(i) label_index_actual];
        if (label_index_actual)~=Test_output(i)+1
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    expect_actual_test;
    TestingAccuracy=1-MissClassificationRate_Testing/nTestingData;