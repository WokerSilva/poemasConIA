\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{config-wokSH}[2024/01/26 WokerSH]
\RequirePackage[utf8]{inputenc}

\LoadClass[12pt, a4paper]{article}

\RequirePackage{fancybox}

%---------------------------------/
%--------------------------------/
\usepackage[                 %   |
  headheight=15pt,  %            |  
  left =2cm,  %  < 1 >           |
  right =2cm, %  < 1 >           | MARGENES DE LA PAGINA
  top =2cm,     %  < 1.5 >       |
  bottom =1.5cm %  < 1.5 >       |
]{geometry}     %                |
%--------------------------------\
%---------------------------------\


%-------------------------------------------------------------|
%-------------Lista de estilos códigos------------------------|
\RequirePackage{listings}

%//////////////// ESTILO JAVA ------------------------------|
\lstdefinestyle{mystylejava}{
    keywordstyle=[1]\color{codeclassjava},
    backgroundcolor=\color{backcolour},
    commentstyle=\color{codegreen},
    numberstyle=\tiny\color{codenumizq},
    stringstyle=\color{codestringjava},
    basicstyle=\footnotesize\ttfamily\color{white},
    breakatwhitespace=false,
    breaklines=true,
    captionpos=b,
    keepspaces=true,
    numbers=left,
    numbersep=5pt,
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=2,            
    otherkeywords={print},
    escapechar=\&,
    literate=                
        {;}{{\textcolor{yellowpoint}{;}}}1
        {+}{{\textcolor{yellowpoint}{+}}}1
        {-}{{\textcolor{yellowpoint}{-}}}1
        {\{}{{\textcolor{yellowpoint}{\{}}}1
        {\}}{{\textcolor{yellowpoint}{\}}}}1
        {[}{{\textcolor{yellowpoint}{[}}}1
        {]}{{\textcolor{yellowpoint}{]}}}1
        {=}{{\textcolor{yellowpoint}{=}}}1
    %frame=shadowbox
}
%//////////////// ESTILO JAVA ------------------------------|
%  % %  %%  %%  %%  %%  %%  %%  %%  %%  %%  %%  %%  %%  %%  %
%//////////////// ESTILO PYTHON ----------------------------|
\lstdefinestyle{mystylepython}{
    backgroundcolor=\color{backcolourPY},
    commentstyle=\color{codegreenPY},
    keywordstyle=\color{codeclassPY},
    numberstyle=\tiny\color{codenumizq},
    stringstyle=\color{codestringPY},
    basicstyle=\footnotesize\ttfamily\color{white},
    breakatwhitespace=false,
    breaklines=true,
    captionpos=b,
    keepspaces=true,
    numbers=left,
    numbersep=5pt,
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=2,
    escapechar=\&,
    literate=                
        {;}{{\textcolor{yellowpoint}{;}}}1
        {+}{{\textcolor{yellowpoint}{+}}}1
        {-}{{\textcolor{yellowpoint}{-}}}1
        {\{}{{\textcolor{yellowpoint}{\{}}}1
        {\}}{{\textcolor{yellowpoint}{\}}}}1
        {[}{{\textcolor{yellowpoint}{[}}}1
        {]}{{\textcolor{yellowpoint}{]}}}1
        {=}{{\textcolor{yellowpoint}{=}}}1
}
%//////////////// ESTILO PYTHON ----------------------------|
%  % %  %%  %%  %%  %%  %%  %%  %%  %%  %%  %%  %%  %%  %%  %
%-------------- ESTILO de Código TERMINAL ------------------|
\lstdefinestyle{mystyleTerminal}{
    backgroundcolor=\color{backcolourTerminal},
    commentstyle=\color{codecommentTerminal},
    keywordstyle=\color{codeclassTerminal},
    numberstyle=\tiny\color{backcolourTerminal},
    stringstyle=\color{codestringTerminal},
    basicstyle=\footnotesize\ttfamily\color{codeclassTerminal},
    breakatwhitespace=false,
    breaklines=true,
    captionpos=b,
    keepspaces=true,
    numbers=left,
    numbersep=5pt,
    showspaces=false,
    showstringspaces=false,
    showtabs=false,
    tabsize=4,    
    escapechar=\&,
    literate = 
            {--}{{\textcolor{codecommentTerminal}{--}}}2            
}
% Usar el estilo de código
\lstset{style=mystyleTerminal}
%-------------- ESTILO de Código TERMINAL ------------------|
%-------------------------------------------------------------|
%-------------Lista de estilos códigos------------------------|
%-------------------------------------------------------------|
%\lstset{style=mystyle}


%-------------------------------------------------------------|
%------------- Modificación de items -------------------------|
\RequirePackage{enumitem}

\newlist{myitemize}{itemize}{4}
\setlist[myitemize,1]{label=\textcolor{Contraste4}{$\bullet$}, left=1.6em}
\setlist[myitemize,2]{label=\textcolor{Contraste4}{$\diamond$}, left=1em}
\setlist[myitemize,3]{label=\textcolor{Contraste4}{$\rightarrow$}, left=1em}
\setlist[myitemize,4]{label=\textcolor{Contraste4}{$\blacksquare$}, left=1em}

\endinput

%-------------------------------------------------------------|
%------------- Modificación de items -------------------------|