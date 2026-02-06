# MIT License
# Copyright (c) 2025 Matthew Abbott
#
# Tests for GnnFacadeCuda Julia bindings

using Test
using GnnFacadeCuda

@testset "GnnFacadeCuda" begin

    @testset "GNN Creation" begin
        gnn = GnnFacade(3, 16, 2, 2)
        @test get_feature_size(gnn) == 3
        @test get_hidden_size(gnn) == 16
        @test get_output_size(gnn) == 2
        @test get_num_message_passing_layers(gnn) == 2
    end

    @testset "Graph Creation" begin
        gnn = GnnFacade(3, 16, 2, 2)
        create_empty_graph!(gnn, 5, 3)
        
        @test get_num_nodes(gnn) == 5
        @test is_graph_loaded(gnn)
    end

    @testset "Edge Operations" begin
        gnn = GnnFacade(3, 16, 2, 2)
        create_empty_graph!(gnn, 5, 3)
        
        idx1 = add_edge!(gnn, 0, 1)
        @test idx1 == 0
        
        idx2 = add_edge!(gnn, 1, 2)
        @test idx2 == 1
        
        @test get_num_edges(gnn) == 2
        @test has_edge(gnn, 0, 1)
        @test !has_edge(gnn, 1, 0)  # Directed graph
        
        found_idx = find_edge_index(gnn, 0, 1)
        @test found_idx == 0
        
        not_found = find_edge_index(gnn, 3, 4)
        @test not_found === nothing
    end

    @testset "Node Features" begin
        gnn = GnnFacade(3, 16, 2, 2)
        create_empty_graph!(gnn, 5, 3)
        
        features = Float32[1.0, 0.5, 0.2]
        set_node_features!(gnn, 0, features)
        
        got = get_node_features(gnn, 0)
        @test got !== nothing
        @test length(got) == 3
        @test got ≈ features
        
        # Test single feature access
        set_node_feature!(gnn, 1, 0, 0.75f0)
        @test get_node_feature(gnn, 1, 0) ≈ 0.75f0
    end

    @testset "Prediction" begin
        gnn = GnnFacade(3, 16, 2, 2)
        create_empty_graph!(gnn, 5, 3)
        add_edge!(gnn, 0, 1)
        add_edge!(gnn, 1, 2)
        
        set_node_features!(gnn, 0, Float32[1.0, 0.5, 0.2])
        set_node_features!(gnn, 1, Float32[0.8, 0.3, 0.1])
        
        prediction = predict!(gnn)
        @test length(prediction) == 2
    end

    @testset "Training" begin
        gnn = GnnFacade(3, 16, 2, 2)
        create_empty_graph!(gnn, 5, 3)
        add_edge!(gnn, 0, 1)
        add_edge!(gnn, 1, 2)
        
        set_node_features!(gnn, 0, Float32[1.0, 0.5, 0.2])
        set_node_features!(gnn, 1, Float32[0.8, 0.3, 0.1])
        
        target = Float32[0.5, 0.5]
        loss = train!(gnn, target)
        @test loss >= 0
    end

    @testset "Save/Load" begin
        tmpfile = tempname() * ".bin"
        
        try
            # Create and save
            gnn1 = GnnFacade(3, 16, 2, 2)
            set_learning_rate!(gnn1, 0.05f0)
            save_model(gnn1, tmpfile)
            
            # Load and verify
            gnn2 = load_gnn(tmpfile)
            @test get_feature_size(gnn2) == 3
            @test get_hidden_size(gnn2) == 16
            @test get_output_size(gnn2) == 2
            
            # Read header without loading
            header = read_model_header(tmpfile)
            @test header.feature_size == 3
            @test header.hidden_size == 16
            @test header.output_size == 2
        finally
            rm(tmpfile, force=true)
        end
    end

    @testset "PageRank" begin
        gnn = GnnFacade(3, 16, 2, 2)
        create_empty_graph!(gnn, 5, 3)
        add_edge!(gnn, 0, 1)
        add_edge!(gnn, 1, 2)
        add_edge!(gnn, 2, 3)
        add_edge!(gnn, 3, 4)
        add_edge!(gnn, 4, 0)
        
        scores = compute_page_rank(gnn)
        @test length(scores) == 5
        
        # Sum should be approximately 1
        @test sum(scores) ≈ 1.0 atol=0.02
    end

    @testset "Masking" begin
        gnn = GnnFacade(3, 16, 2, 2)
        create_empty_graph!(gnn, 5, 3)
        
        # All nodes should be active by default
        @test get_masked_node_count(gnn) == 5
        
        # Mask a node
        set_node_mask!(gnn, 0, false)
        @test !get_node_mask(gnn, 0)
        @test get_masked_node_count(gnn) == 4
        
        # Unmask
        set_node_mask!(gnn, 0, true)
        @test get_node_mask(gnn, 0)
        @test get_masked_node_count(gnn) == 5
    end

    @testset "Analytics" begin
        gnn = GnnFacade(3, 16, 2, 2)
        
        # Parameter count should be positive
        @test get_parameter_count(gnn) > 0
        
        # Architecture summary should not be empty
        summary = get_architecture_summary(gnn)
        @test !isempty(summary)
        
        # Gradient flow info
        info = get_gradient_flow(gnn, 0)
        @test info isa GradientFlowInfo
    end

    @testset "JSON Export" begin
        gnn = GnnFacade(3, 16, 2, 2)
        create_empty_graph!(gnn, 3, 3)
        add_edge!(gnn, 0, 1)
        add_edge!(gnn, 1, 2)
        
        json = export_graph_to_json(gnn)
        @test !isempty(json)
        @test startswith(json, "{")
    end

    @testset "Display" begin
        gnn = GnnFacade(3, 16, 2, 2)
        create_empty_graph!(gnn, 5, 3)
        
        # Should not throw
        str = sprint(show, gnn)
        @test contains(str, "GnnFacade")
        @test contains(str, "feature_size=3")
    end

end
