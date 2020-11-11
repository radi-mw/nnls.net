using System;
using System.Linq;
using NnlsNet;

namespace NnlsNet.Test
{
    class Program
    {
        static void Main(string[] args)
        {

            var x1 = Enumerable.Range(0, 100).Select(i => (double)i).ToArray();
            var x2 = Enumerable.Range(0, 100).Select(i => 10 * Math.Sin(i * Math.PI /10)).ToArray();
            var x2mean = x2.Average();

            var y1 = x1.Zip(x2, (a, b) => ( a + b - x2mean)).ToArray();
            var y2 = x1.Zip(x2, (a, b) => (a - b + x2mean)).ToArray();

            var beta12 = Nnls.Solve(new double[][] { x1, x2 }, y1);
            var beta34 = Nnls.Solve(new double[][] { x1, x2 }, y2);

            Console.WriteLine($"beta1: {beta12[0]}");
            Console.WriteLine($"beta2: {beta12[1]}");
            Console.WriteLine($"beta3: {beta34[0]}");
            Console.WriteLine($"beta4: {beta34[1]}");
        }
    }
}
