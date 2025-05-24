Lexer lexer("+/ i. 5"); auto tokens = lexer.tokenize(); for(auto& t : tokens) std::cout << t << std::endl;
