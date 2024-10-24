function [RBF] = SelectRBF(N, Tini, BasisStructure, n_basisKoopman, n_basisNARX, Basis_funcKoopman, Basis_funcNARX)
%Select RBF
%   Select the appropriate RBF for different basis structures

%Construct String of Basis
if  strcmp(BasisStructure, 'Koopman')
    file_string = 'RBFs/'+string(Basis_funcKoopman)+'/RBF_Params_Koopman_Tini'+string(Tini)+'_nbasisKoopman'+string(n_basisKoopman)+'_N'+string(N)+'_KoopmanBasis_'+string(Basis_funcKoopman)+'.mat';
    RBF = load(file_string)
    
    %Add Basis Function Type
    RBF.Basis_funcKoopman = Basis_funcKoopman;
    RBF.Basis_funcNARX = "";

    %Add Parameter String
    RBF.ParameterString = '_N'+string(N)+'_n_basisKoopman'+string(n_basisKoopman)+'_Tini'+string(Tini)+'_RBFTypeKoopman_'+Basis_funcKoopman;

elseif  strcmp(BasisStructure,'KoopmanResnet')
    file_string = 'RBFs/'+string(Basis_funcKoopman)+'/RBF_Params_KoopmanResnet_Tini'+string(Tini)+'_nbasisKoopman'+string(n_basisKoopman)+'_N'+string(N)+'_KoopmanBasis_'+string(Basis_funcKoopman)+'.mat';
    RBF = load(file_string)
    
    %Add Basis Function Type
    RBF.Basis_funcKoopman = Basis_funcKoopman;
    RBF.Basis_funcNARX = "";

    %Add Parameter String
    RBF.ParameterString = '_N'+string(N)+'_n_basisKoopman'+string(n_basisKoopman)+'_Tini'+string(Tini)+'_RBFTypeKoopman_'+Basis_funcKoopman;

elseif  strcmp(BasisStructure, 'NARX')
    file_string = 'RBFs/'+string(Basis_funcNARX)+'/RBF_Params_NARX_Tini'+string(Tini)+'_nbasisNARX'+string(n_basisNARX)+'_N'+string(N)+'_NARXBasis_'+string(Basis_funcNARX)+'.mat';
    RBF = load(file_string)
    
    %Add Basis Function Type
    RBF.Basis_funcKoopman = "";
    RBF.Basis_funcNARX = Basis_funcNARX;

    %Add Parameter String
    RBF.ParameterString = '_N'+string(N)+'_n_basisNARX'+string(n_basisNARX)+'_Tini'+string(Tini)+'_RBFTypeNARX_'+Basis_funcNARX;

elseif  strcmp(BasisStructure, 'NARXResnet')
    file_string = 'RBFs/'+string(Basis_funcNARX)+'/RBF_Params_'+BasisStructure+'_Tini'+string(Tini)+'_nbasisNARX'+string(n_basisNARX)+'_N'+string(N)+'_NARXBasis_'+string(Basis_funNARX)+'.mat';
    RBF = load(file_string)

    %Add Basis Function Type
    RBF.Basis_funcKoopman = "";
    RBF.Basis_funcNARX = Basis_funcNARX;


    %Add Parameter String
    RBF.ParameterString = '_N'+string(N)+'_n_basisNARX'+string(n_basisNARX)+'_Tini'+string(Tini)+'_RBFTypeNARX_'+Basis_funcNARX;

elseif  strcmp(BasisStructure, 'KoopmanNARX')
    file_string = 'RBFs/'+string(Basis_funcKoopman)+'/'+string(Basis_funcNARX)+ '/RBF_Params_'+BasisStructure+'_Tini'+string(Tini)+'_nbasisKoopman'+string(n_basisKoopman)+'_nbasisFullBasis'+string(n_basisNARX)+'_N'+string(N)+'_KoopmanBasis_'+string(Basis_funcKoopman)+'_FullBasis_'+string(Basis_funcNARX)+'.mat';
    RBF = load(file_string)

    %Add Basis Function Type
    RBF.Basis_funcKoopman = Basis_funcKoopman;
    RBF.Basis_funcNARX = Basis_funcNARX;


    %Add Parameter String
    RBF.ParameterString = '_N'+string(N)+'_n_basisKoopman'+string(n_basisKoopman)+'_n_basisNARX'+string(n_basisNARX)+'_Tini'+string(Tini)+'_RBFTypeKoopman_'+Basis_funcKoopman+'_RBFTypeNARX_'+Basis_funcNARX;
end

%Load the Basis Function


%Add Model Parameters
RBF.N = N;
RBF.Tini = Tini;

%Add Basis Structure String
RBF.BasisStructure = BasisStructure;

RBF.data_mean = double(RBF.data_mean);
RBF.data_std = double(RBF.data_std);

end