
using System;
using System.Collections.Generic;
using System.Linq;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NnlsNet
{
    public static class Nnls
    {
        public static double[] Solve(IEnumerable<double>[] A, IEnumerable<double> b, IEnumerable<double> wt = null, double tol = 1e-6)
        {
            List<int> pInd = new List<int>();
            List<int> rInd = new List<int>();
            for (int i = 0; i < A.Length; i++)
            {
                rInd.Add(i);
            }
            double[] x = new double[A.Length];
            Matrix<double> matA = DenseMatrix.OfColumns(A);
            Vector<double> vecb = DenseVector.OfEnumerable(b);
            Matrix<double> matW = null;
            if (wt != null)
            {
                var cnt = wt.Count();
                matW = DiagonalMatrix.OfDiagonal(cnt, cnt, wt);
            }
            else
            {
                matW = DenseMatrix.CreateIdentity(b.Count());
            }

            Matrix<double> ATA = (matA.TransposeThisAndMultiply(matW)).Multiply(matA);
            Vector<double> ATb = (matA.TransposeThisAndMultiply(matW)).Multiply(vecb);

            double[] res = (ATb - (ATA).Multiply(DenseVector.OfEnumerable(x))).ToArray();

            while (rInd.Count > 0 && rInd.Max(r => res[r]) > tol)
            {
                //loop B, Build SP

                var pi = rInd.ArgMax(r => res[r]);
                rInd.Remove(pi);
                pInd.Add(pi);

                var SP = GetSP(ATA, ATb, pInd);
                var s = GetS(SP, A.Length, pInd);
                while (SP.Min() <= 0)
                {
                    //loop C
                    var alpha = -pInd.Select((v, index) => new { V = v, X = index }).Min(p => x[p.V] / (x[p.V] - SP[p.X]));
                    for (int i = 0; i < x.Length; i++)
                    {
                        x[i] = x[i] + (s[i] - x[i]) * alpha;

                    }

                    res = (ATb - (ATA).Multiply(DenseVector.OfEnumerable(x))).ToArray();

                    rInd.Clear();
                    pInd.Clear();
                    for (int i = 0; i < res.Length; i++)
                    {
                        if (res[i] <= tol)
                        {
                            rInd.Add(i);
                        }
                        else
                        {
                            pInd.Add(i);
                        }
                    }

                    SP = GetSP(ATA, ATb, pInd);
                    s = GetS(SP, A.Length, pInd);
                }

                x = s;
                res = (ATb - (ATA).Multiply(DenseVector.OfEnumerable(x))).ToArray();

            }

            return x;
        }
        
        static Vector<double> GetSP(Matrix<double> ATA, Vector<double> ATb, List<int> pInd)
        {
            var arrATA = ATA.ToArray();
            var arrATAP = new double[pInd.Count, pInd.Count];
            for (int i = 0; i < pInd.Count; i++)
            {
                for (int j = 0; j < pInd.Count; j++)
                {
                    arrATAP[i, j] = ATA[pInd[i], pInd[j]];
                }
            }
            var ATAP = DenseMatrix.OfArray(arrATAP);
            var arrATb = ATb.ToArray();
            var arrATbP = new double[pInd.Count];
            for (int i = 0; i < pInd.Count; i++)
            {
                arrATbP[i] = ATb[pInd[i]];
            }
            var ATbP = DenseVector.OfArray(arrATbP);
            var SP = ATAP.QR().Solve(ATbP);
            return SP;
        }
        
        static double[] GetS(Vector<double> SP, int n, List<int> pInd)
        {
            double[] s = new double[n];
            for (int i = 0; i < pInd.Count; i++)
            {
                s[pInd[i]] = SP[i];
            }
            return s;
        }
    }



    public static class EnumerableExtensions
    {
        public static TSrc ArgMax<TSrc, TArg>(this IEnumerable<TSrc> ie, Converter<TSrc, TArg> fn) where TArg : IComparable<TArg>
        {
            IEnumerator<TSrc> e = ie.GetEnumerator();
            if (!e.MoveNext())
                throw new InvalidOperationException("Sequence has no elements.");

            TSrc t_try, t = e.Current;
            if (!e.MoveNext())
                return t;

            TArg v, max_val = fn(t);
            do
            {
                if ((v = fn(t_try = e.Current)).CompareTo(max_val) > 0)
                {
                    t = t_try;
                    max_val = v;
                }
            }
            while (e.MoveNext());
            return t;
        }

    }
}
