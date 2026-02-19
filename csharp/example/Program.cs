/**
 * @file
 * @ingroup GNN_Wrappers
 */
using System;
using GnnFacadeCuda;

class Program
{
    static void Main(string[] args)
    {
        try
        {
            using var gnn = new GnnFacade(3, 16, 2, 2);

            Console.WriteLine($"Backend: {gnn.BackendName}");
            Console.WriteLine($"Parameters: {gnn.ParameterCount}");

            gnn.CreateEmptyGraph(5, 3);
            gnn.AddEdge(0, 1);
            gnn.AddEdge(1, 2);
            gnn.AddEdge(2, 3);
            gnn.AddEdge(3, 4);

            gnn.SetNodeFeatures(0, new float[] { 1.0f, 0.5f, 0.2f });
            gnn.SetNodeFeatures(1, new float[] { 0.3f, 0.8f, 0.1f });
            gnn.SetNodeFeatures(2, new float[] { 0.7f, 0.2f, 0.9f });
            gnn.SetNodeFeatures(3, new float[] { 0.4f, 0.6f, 0.3f });
            gnn.SetNodeFeatures(4, new float[] { 0.9f, 0.1f, 0.5f });

            Console.WriteLine($"Nodes: {gnn.NumNodes}, Edges: {gnn.NumEdges}");

            float[] prediction = gnn.Predict();
            Console.Write("Prediction: [");
            Console.Write(string.Join(", ", prediction));
            Console.WriteLine("]");

            float[] target = { 0.5f, 0.5f };
            for (uint i = 0; i < 10; i++)
            {
                float loss = gnn.Train(target);
                if (i % 5 == 0)
                    Console.WriteLine($"Epoch {i}: loss = {loss}");
            }

            float[] pageRank = gnn.ComputePageRank();
            Console.Write("PageRank: [");
            Console.Write(string.Join(", ", pageRank));
            Console.WriteLine("]");

            Console.WriteLine(gnn.GetArchitectureSummary());
        }
        catch (GnnException ex)
        {
            Console.Error.WriteLine($"GNN Error: {ex.Message}");
            Environment.Exit(1);
        }
    }
}
